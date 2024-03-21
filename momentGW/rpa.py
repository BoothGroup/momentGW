"""
Construct RPA moments.
"""

import numpy as np
import scipy.optimize

from momentGW import dTDA, logging, mpi_helper, util


class dRPA(dTDA):
    """
    Compute the self-energy moments using dRPA and numerical integration.

    Parameters
    ----------
    gw : BaseGW
        GW object.
    nmom_max : int
        Maximum moment number to calculate.
    integrals : Integrals
        Density-fitted integrals.
    mo_energy : dict, optional
        Molecular orbital energies. Keys are "g" and "w" for the Green's
        function and screened Coulomb interaction, respectively.
        If `None`, use `gw.mo_energy` for both. Default value is `None`.
    mo_occ : dict, optional
        Molecular orbital occupancies. Keys are "g" and "w" for the
        Green's function and screened Coulomb interaction, respectively.
        If `None`, use `gw.mo_occ` for both. Default value is `None`.

    Notes
    -----
    See `momentGW.tda.dTDA.__init__` for initialisation details and
    `momentGW.tda.dTDA.kernel` for calculation run details.
    """

    @logging.with_timer("Numerical integration")
    @logging.with_status("Performing numerical integration")
    def integrate(self):
        """Optimise the quadrature and perform the integration for the
        zeroth moment.

        Returns
        -------
        integral : numpy.ndarray
            Integral array, including the offset part.
        """

        p0, p1 = self.mpi_slice(self.nov)

        # Construct energy differences
        d_full = util.build_1h1p_energies(self.mo_energy_w, self.mo_occ_w).ravel()
        d = d_full[p0:p1]

        # Calculate diagonal part of ERI
        diag_eri = np.zeros((self.nov,))
        diag_eri[p0:p1] = util.einsum("np,np->p", self.integrals.Lia, self.integrals.Lia)
        diag_eri = mpi_helper.allreduce(diag_eri)

        # Get the offset integral quadrature
        quad = self.optimise_offset_quad(d_full, diag_eri)

        # Perform the offset integral
        offset = self.eval_offset_integral(quad, d)

        # Get the main integral quadrature
        quad = self.optimise_main_quad(d_full, diag_eri)

        # Perform the main integral
        integral = self.eval_main_integral(quad, d)

        # Report quadrature error
        if self.report_quadrature_error:
            a = np.sum((integral[0] - integral[2]) ** 2)
            b = np.sum((integral[0] - integral[1]) ** 2)
            a, b = mpi_helper.allreduce(np.array([a, b]))
            a, b = a**0.5, b**0.5
            err = self.estimate_error_clencur(a, b)
            style_half = logging.rate(a, 1e-4, 1e-3)
            style_quar = logging.rate(b, 1e-8, 1e-6)
            style_full = logging.rate(err, 1e-12, 1e-9)
            logging.write(
                f"Error in integral:  [{style_full}]{err:.3e}[/] "
                f"(half = [{style_half}]{a:.3e}[/], quarter = [{style_quar}]{b:.3e}[/])",
            )

        return integral[0] + offset

    @logging.with_timer("Density-density moments")
    @logging.with_status("Constructing density-density moments")
    def build_dd_moments(self, integral=None):
        """Build the moments of the density-density response.

        Parameters
        ----------
        integral : numpy.ndarray, optional
            Integral array, including the offset part. If `None`,
            calculate from scratch. Default is `None`.

        Returns
        -------
        moments : numpy.ndarray
            Moments of the density-density response.
        """

        if integral is None:
            integral = self.integrate()

        p0, p1 = self.mpi_slice(self.nov)
        moments = np.zeros((self.nmom_max + 1, self.naux, p1 - p0))

        # Construct energy differences
        d_full = util.build_1h1p_energies(self.mo_energy_w, self.mo_occ_w).ravel()
        d = d_full[p0:p1]

        # Calculate (L|ia) D_{ia} and (L|ia) D_{ia}^{-1} intermediates
        Liad = self.integrals.Lia * d[None]
        Liadinv = self.integrals.Lia / d[None]

        # Construct (A-B)^{-1}
        u = np.dot(Liadinv, self.integrals.Lia.T) * 4.0  # aux^2 o v
        u = mpi_helper.allreduce(u)
        u = np.linalg.inv(np.eye(self.naux) + u)

        # Get the zeroth order moment
        moments[0] = integral / d[None]
        tmp = np.linalg.multi_dot((integral, Liadinv.T, u))  # aux^2 o v
        tmp = mpi_helper.allreduce(tmp)
        moments[0] -= np.dot(tmp, Liadinv) * 4.0  # aux^2 o v
        del u, tmp

        # Get the first order moment
        moments[1] = Liad

        # Get the higher order moments
        for i in range(2, self.nmom_max + 1):
            moments[i] = moments[i - 2] * d[None] ** 2
            tmp = np.dot(moments[i - 2], self.integrals.Lia.T)  # aux^2 o v
            tmp = mpi_helper.allreduce(tmp)
            moments[i] += np.dot(tmp, Liad) * 4.0  # aux^2 o v
            del tmp

        return moments

    @logging.with_timer("Density-density moments")
    @logging.with_status("Constructing density-density moments")
    def build_dd_moments_exact(self):
        """Build the exact moments of the density-density response.

        Returns
        -------
        moments : numpy.ndarray
            Moments of the density-density response.
        """

        import sys

        sys.argv.append("--silent")
        from vayesta.rpa import ssRPA

        rpa = ssRPA(self.gw._scf)
        rpa.kernel()

        rot = np.concatenate([self.integrals.Lia, self.integrals.Lia], axis=-1)

        moments = rpa.gen_moms(self.nmom_max)
        moments = util.einsum("nij,Pi->nPj", moments, rot)

        return moments[:, :, : self.nov]

    def build_dp_moments(self):
        """
        Build the moments of the dynamic polarizability for optical
        spectra calculations.

        Returns
        -------
        moments : numpy.ndarray
            Moments of the dynamic polarizability.
        """
        raise NotImplementedError

    # --- Numerical integration functions:

    @staticmethod
    def rescale_quad(bare_quad, a):
        """Rescale quadrature for grid space `a`."""
        return bare_quad[0] * a, bare_quad[1] * a

    def optimise_main_quad(self, d, diag_eri, name="main"):
        """
        Optimise the grid spacing of Clenshaw-Curtis quadrature for the
        main integral.

        Parameters
        ----------
        d : numpy.ndarray
            Array of orbital energy differences.
        diag_eri : numpy.ndarray
            Diagonal of the ERIs.
        name : str, optional
            Name of the integral. Default value is `"main"`.

        Returns
        -------
        points : numpy.ndarray
            The quadrature points.
        weights : numpy.ndarray
            The quadrature weights.
        """

        bare_quad = self.gen_clencur_quad_inf(even=True)

        exact = np.sum((d * (d + diag_eri)) ** 0.5)
        exact -= 0.5 * np.dot(1.0 / d, d * diag_eri)
        exact -= np.sum(d)

        integrand = lambda quad: self.eval_diag_main_integral(quad, d, diag_eri)
        quad = self.get_optimal_quad(bare_quad, integrand, exact, name=name)

        return quad

    def optimise_offset_quad(self, d, diag_eri, name="offset"):
        """
        Optimise the grid spacing of Clenshaw-Curtis quadrature for the
        main integral.

        Parameters
        ----------
        d : numpy.ndarray
            Array of orbital energy differences.
        diag_eri : numpy.ndarray
            Diagonal of the ERIs.
        name : str, optional
            Name of the integral. Default value is `"offset"`.

        Returns
        -------
        points : numpy.ndarray
            The quadrature points.
        weights : numpy.ndarray
            The quadrature weights.
        """

        bare_quad = self.gen_gausslag_quad_semiinf()

        exact = 0.5 * np.dot(1.0 / d, d * diag_eri)

        integrand = lambda quad: self.eval_diag_offset_integral(quad, d, diag_eri)
        quad = self.get_optimal_quad(bare_quad, integrand, exact, name=name)

        return quad

    def get_optimal_quad(self, bare_quad, integrand, exact, name=None):
        """Get the optimal quadrature.

        Parameters
        ----------
        bare_quad : tuple
            The quadrature points and weights.
        integrand : function
            The integrand function.
        exact : float
            The exact value of the integral.
        name : str, optional
            Name of the integral. Default value is `None`.

        Returns
        -------
        points : numpy.ndarray
            The quadrature points.
        weights : numpy.ndarray
            The quadrature weights.
        """

        def diag_err(spacing):
            return np.abs(integrand(self.rescale_quad(bare_quad, 10**spacing)) - exact)

        res = scipy.optimize.minimize_scalar(diag_err, bounds=(-6, 2), method="bounded")
        if not res.success:
            raise RuntimeError("Could not optimise `a` value.")

        solve = 10**res.x
        full_name = f"{f'{name} ' if name else ''}quadrature".capitalize()
        style = logging.rate(res.fun, 1e-14, 1e-10)
        logging.write(f"{full_name} scale:  {solve:.2e} (error = [{style}]{res.fun:.2e}[/])")

        return self.rescale_quad(bare_quad, solve)

    def optimise_offset_quad(self, d, diag_eri):
        """
        Optimise the grid spacing of Gauss-Laguerre quadrature for the
        offset integral.

        Parameters
        ----------
        d : numpy.ndarray
            Orbital energy differences.
        diag_eri : numpy.ndarray
            Diagonal of the ERIs.

        Returns
        -------
        points : numpy.ndarray
            The quadrature points.
        weights : numpy.ndarray
            The quadrature weights.
        """

        bare_quad = self.gen_gausslag_quad_semiinf()
        exact = np.dot(1.0 / d, d * diag_eri)

        integrand = lambda quad: self.eval_diag_offset_integral(quad, d, diag_eri)
        quad = self.get_optimal_quad(bare_quad, integrand, exact)

        return quad

    def eval_diag_offset_integral(self, quad, d, diag_eri):
        """Evaluate the diagonal of the offset integral.

        Parameters
        ----------
        quad : tuple
            The quadrature points and weights.
        d : numpy.ndarray
            Orbital energy differences.
        diag_eri : numpy.ndarray
            Diagonal of the ERIs.

        Returns
        -------
        integral : numpy.ndarray
            Offset integral.
        """

        # TODO check this: is this still right, does it need a factor 4.0?

        integral = 0.0

        for point, weight in zip(*quad):
            expval = np.exp(-2 * point * d)
            res = np.dot(expval, d * diag_eri)
            integral += 2 * res * weight  # aux^2 o v
        return integral

    def eval_offset_integral(self, quad, d, Lia=None):
        """Evaluate the offset integral.

        Parameters
        ----------
        quad : tuple
            The quadrature points and weights.
        d : numpy.ndarray
            Orbital energy differences.
        Lia : numpy.ndarray
            The (aux, W occ, W vir) integral array. If `None`, use
            `self.integrals.Lia`. Keyword argument allows for the use of
            this function with `uhf` and `pbc` modules.

        Returns
        -------
        integral : numpy.ndarray
            Offset integral.
        """

        if Lia is None:
            Lia = self.integrals.Lia

        Liad = Lia * d[None]
        integral = np.zeros_like(Liad)

        for point, weight in zip(*quad):
            expval = np.exp(-point * d)
            lhs = np.dot(Liad * expval[None], Lia.T)  # aux^2 o v
            lhs = mpi_helper.allreduce(lhs)
            rhs = Lia * expval[None]  # aux o v
            res = np.dot(lhs, rhs)
            integral += res * weight  # aux^2 o v

        integral *= 4.0
        integral += Liad

        return integral

    def optimise_main_quad(self, d, diag_eri):
        """
        Optimise the grid spacing of Clenshaw-Curtis quadrature for the
        main integral.

        Parameters
        ----------
        d : numpy.ndarray
            Orbital energy differences.
        diag_eri : numpy.ndarray
            Diagonal of the ERIs.

        Returns
        -------
        points : numpy.ndarray
            The quadrature points.
        weights : numpy.ndarray
            The quadrature weights.
        """

        bare_quad = self.gen_clencur_quad_inf(even=True)

        exact = np.sum((d * (d + diag_eri)) ** 0.5)
        exact -= 0.5 * np.dot(1.0 / d, d * diag_eri)
        exact -= np.sum(d)

        integrand = lambda quad: self.eval_diag_main_integral(quad, d, diag_eri)
        quad = self.get_optimal_quad(bare_quad, integrand, exact)

        return quad

    def eval_diag_main_integral(self, quad, d, diag_eri):
        """Evaluate the diagonal of the main integral.

        Parameters
        ----------
        quad : tuple
            The quadrature points and weights.
        d : numpy.ndarray
            Orbital energy differences.
        diag_eri : numpy.ndarray
            Diagonal of the ERIs.

        Returns
        -------
        integral : numpy.ndarray
            Main integral.
        """

        integral = 0.0

        def diag_contrib(x, freq):
            integral = np.ones_like(x)
            integral -= freq**2 / (x + freq**2)
            integral /= np.pi
            return integral

        for point, weight in zip(*quad):
            f = 1.0 / (d**2 + point**2)

            contrib = diag_contrib(d * (d + diag_eri), point)
            contrib -= diag_contrib(d**2, point)
            contrib = np.sum(contrib)
            contrib -= point**2 * np.dot(f**2, d * diag_eri) / np.pi

            integral += weight * contrib

        return integral

    def eval_main_integral(self, quad, d, Lia=None):
        """Evaluate the main integral.

        Parameters
        ----------
        quad : tuple
            The quadrature points and weights.
        d : numpy.ndarray
            Orbital energy differences.
        Lia : numpy.ndarray
            The (aux, W occ, W vir) integral array. If `None`, use
            `self.integrals.Lia`. Keyword argument allows for the use of
            this function with `uhf` and `pbc` modules.

        Returns
        -------
        integral : numpy.ndarray
            Offset integral.
        """

        if Lia is None:
            Lia = self.integrals.Lia

        naux, nov = Lia.shape  # This `nov` is actually self.mpi_size(nov)
        dim = 3 if self.report_quadrature_error else 1
        Liad = Lia * d[None]
        integral = np.zeros((dim, naux, nov))

        for i, (point, weight) in enumerate(zip(*quad)):
            f = 1.0 / (d**2 + point**2)
            q = np.dot(Lia * f[None], Liad.T) * 4.0  # aux^2 o v
            q = mpi_helper.allreduce(q)
            tmp = np.linalg.inv(np.eye(naux) + q) - np.eye(naux)

            contrib = np.linalg.multi_dot((q, tmp, Lia))  # aux^2 o v
            contrib = weight * (contrib * f[None] * (point**2 / np.pi))

            integral[0] += contrib
            if i % 2 == 0 and self.report_quadrature_error:
                integral[1] += 2 * contrib
            if i % 4 == 0 and self.report_quadrature_error:
                integral[2] += 4 * contrib

        return integral

    def gen_clencur_quad_inf(self, even=False):
        """
        Generate quadrature points and weights for Clenshaw-Curtis
        quadrature over an (-inf, +inf).

        Parameters
        ----------
        even : bool, optional
            Whether to assume an even grid. Default is `False`.

        Returns
        -------
        points : numpy.ndarray
            Quadrature points.
        weights : numpy.ndarray
            Quadrature weights.
        """

        factor = 1 + int(even)
        tvals = np.arange(1, self.gw.npoints + 1) / self.gw.npoints
        tvals *= np.pi / factor

        points = 1.0 / np.tan(tvals)
        weights = np.pi * factor / (2 * self.gw.npoints * np.sin(tvals) ** 2)
        if even:
            weights[-1] /= 2

        return points, weights

    def gen_gausslag_quad_semiinf(self):
        """
        Generate quadrature points and weights for Gauss-Laguerre
        quadrature over an (0, +inf).

        Returns
        -------
        points : numpy.ndarray
            Quadrature points.
        weights : numpy.ndarray
            Quadrature weights.
        """

        points, weights = np.polynomial.laguerre.laggauss(self.gw.npoints)
        weights *= np.exp(points)

        return points, weights

    def estimate_error_clencur(self, i4, i2, imag_tol=1e-10):
        """
        Estimate the quadrature error for Clenshaw-Curtis quadrature.

        Parameters
        ----------
        i4 : numpy.ndarray
            Integral at one-quarter the number of points.
        i2 : numpy.ndarray
            Integral at one-half the number of points.
        imag_tol : float, optional
            Threshold to consider the imaginary part of a root to be zero.
            Default value is `1e-10`.

        Returns
        -------
        error : numpy.ndarray
            Estimated error.
        """

        if (i4 - i2) < 1e-14:
            return 0.0

        # Eq. 103 from https://arxiv.org/abs/2301.09107
        roots = np.roots([1, 0, i4 / (i4 - i2), -i2 / (i4 - i2)])

        # Require a real root between 0 and 1
        real_roots = roots[np.abs(roots.imag) < 1e-10].real

        # Check how many there are
        if len(real_roots) > 1:
            logging.warn(
                "Nested quadrature error estimation gives [bad]%d real roots[/]. "
                "Taking smallest positive root." % len(real_roots),
            )
        else:
            logging.write(
                f"Nested quadrature error estimation gives {len(real_roots)} "
                f"real root{'s' if len(real_roots) != 1 else ''}.",
            )

        # Check if there is a root between 0 and 1
        if not np.any(np.logical_and(real_roots > 0, real_roots < 1)):
            logging.warn(
                "Nested quadrature error estimation gives [bad]no root between 0 and 1[/]."
            )
            return np.nan
        else:
            root = np.min(real_roots[np.logical_and(real_roots > 0, real_roots < 1)])

        # Calculate the error
        error = i2 / (1.0 + root**-2)

        return error
