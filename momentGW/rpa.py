"""Construct RPA moments."""

import numpy as np
import scipy.optimize

from momentGW import logging, mpi_helper, util
from momentGW.tda import dTDA


class dRPA(dTDA):
    """Compute the self-energy moments using dRPA and numerical integration.

    Parameters
    ----------
    gw : BaseGW
        GW object.
    nmom_max : int
        Maximum moment number to calculate.
    integrals : BaseIntegrals
        Integrals object.
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
    def build_zeroth_moment(self, m0=None):
        """Build the zeroth moment by optimising the quadrature and perform the integration for
        the zeroth moment.

        Returns
        -------
        zeroth moment : numpy.ndarray
            Zeroth moment of the density-density response.
        """

        p0, p1 = self.mpi_slice(self.nov)

        # Construct energy differences
        d_full = util.build_1h1p_energies(self.mo_energy_w, self.mo_occ_w).ravel()

        # Calculate diagonal part of ERI
        diag_eri = np.zeros((self.nov,))
        diag_eri[p0:p1] = util.einsum("np,np->p", self.integrals.Lia, self.integrals.Lia)
        diag_eri = mpi_helper.allreduce(diag_eri)

        # Get the main integral quadrature
        quad = self.optimise_main_quad(d_full, diag_eri)

        # Perform the main integral
        integral = self.eval_main_integral(quad)

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
            integral = np.delete(integral, [1, 2], 0)
        return integral[0]

    @logging.with_timer("Nth density-density moments")
    @logging.with_status("Constructing nth density-density moment")
    def build_nth_dd_moment(self, n, recursion_term=None, zeroth_mom=None):
        """Build the nth moment of the density-density response.

        Parameters
        ----------
        n : int
            Moment order to be built.
        recursion_term : numpy.ndarray, optional
            Previous recursion term required to build the next moment. In the case of RPA this is
            the appropriate [(A+B)(A-B)]^(n-2/2) for the nth moment. These are only calculated on
            even moments, odd moments use the previous even moment value.
        zeroth moment : numpy.ndarray, optional
            Zeroth moment of the density-density response.

        Returns
        -------
        recursion_term : numpy.ndarray
            Term required for the next moment. In the case of RPA this is [(A+B)(A-B)]^(n/2)
        eta_aux : numpy.ndarray
            The nth density-density response moment in (N_aux,N_aux) form
        """
        if n % 2 == 0:
            if zeroth_mom is None:
                zeroth_mom = self.build_zeroth_moment()
            if n != 0:
                tmp = np.dot(self.integrals.Lia * self.d[None], recursion_term) * 4.0
                tmp = mpi_helper.allreduce(tmp)
                recursion_term = util.einsum("i, iP->iP", self.d**2, recursion_term)
                recursion_term += util.einsum("Pi,PQ->iQ", self.integrals.Lia, tmp)
                del tmp
            elif n == 0 and recursion_term is None:
                recursion_term = self.integrals.Lia.T
            return recursion_term, np.dot(zeroth_mom, recursion_term)

        else:
            if recursion_term is None:
                raise AttributeError(
                    f"To build the {n}th dd-moment, a recursion_term must be provided"
                )
            return recursion_term, np.dot(self.integrals.Lia * self.d[None], recursion_term)

    @logging.with_timer("Density-density moments")
    @logging.with_status("Constructing density-density moments")
    def build_dd_moments(self, integral=None):
        """Build the moments of the density-density response.

        Parameters
        ----------
        integral : numpy.ndarray, optional
            Integral array. If `None`, calculate from scratch. Default is `None`.

        Returns
        -------
        moments : numpy.ndarray
            Moments of the density-density response.
        """
        if self.d is None:
            self._build_d()

        if integral is None:
            integral = self.build_zeroth_moment()

        p0, p1 = self.mpi_slice(self.nov)
        moments = np.zeros((self.nmom_max + 1, self.naux, p1 - p0))

        # Construct energy differences
        d_full = util.build_1h1p_energies(self.mo_energy_w, self.mo_occ_w).ravel()
        d = d_full[p0:p1]

        # Get the zeroth order moment
        moments[0] = integral

        # Get the first order moment
        moments[1] = self.integrals.Lia * d[None]

        # Get the higher order moments
        for i in range(2, self.nmom_max + 1):
            moments[i] = moments[i - 2] * d[None] ** 2
            tmp = np.dot(moments[i - 2], self.integrals.Lia.T)  # aux^2 o v
            tmp = mpi_helper.allreduce(tmp)
            moments[i] += np.dot(tmp, moments[1]) * 4.0  # aux^2 o v
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
        """Build the moments of the dynamic polarizability for optical spectra calculations.

        Returns
        -------
        moments : numpy.ndarray
            Moments of the dynamic polarizability.
        """
        raise NotImplementedError

    # --- Numerical integration functions:

    @staticmethod
    def rescale_quad(bare_quad, a):
        """Rescale quadrature for grid space `a`.

        Parameters
        ----------
        bare_quad : tuple
            The quadrature points and weights.
        a : float
            Grid spacing.

        Returns
        -------
        points : numpy.ndarray
            The quadrature points.
        weights : numpy.ndarray
            The quadrature weights.
        """
        return bare_quad[0] * a, bare_quad[1] * a

    def optimise_main_quad(self, d, diag_eri, name="main"):
        """Optimise the grid spacing of Clenshaw-Curtis quadrature for the main integral.

        Parameters
        ----------
        d : numpy.ndarray
            Orbital energy differences.
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

        # Generate the bare quadrature
        bare_quad = self.gen_ClenCur_quad_semiinf()

        # Calculate the exact value of the integral for the diagonal
        exact = np.sum(d * (d * (d + diag_eri)) ** -0.5)

        # Define the integrand
        integrand = lambda quad: self.eval_diag_main_integral(quad, d, diag_eri)

        # Get the optimal quadrature
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
            """Calculate the error in the diagonal integral."""
            return np.abs(integrand(self.rescale_quad(bare_quad, 10**spacing)) - exact)

        # Optimise the grid spacing
        res = scipy.optimize.minimize_scalar(diag_err, bounds=(-2, 4), method="bounded")
        if not res.success:
            raise RuntimeError("Could not optimise `a` value.")

        # Get the scale
        solve = 10**res.x

        # Report the result
        full_name = f"{f'{name} ' if name else ''}quadrature".capitalize()
        style = logging.rate(res.fun, 1e-14, 1e-10)
        logging.write(f"{full_name} scale:  {solve:.2e} (error = [{style}]{res.fun:.2e}[/])")

        return self.rescale_quad(bare_quad, solve)

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

        for point, weight in zip(*quad):
            contrib = (d + diag_eri) * d + point**2
            contrib = np.sum(d * contrib ** (-1))

            integral += weight * contrib * 2 / np.pi

        return integral

    def eval_main_integral(self, quad, Lia=None):
        """Evaluate the main integral.

        Parameters
        ----------
        quad : tuple
            The quadrature points and weights.
        Lia : numpy.ndarray
            The (aux, W occ, W vir) integral array. If `None`, use
            `self.integrals.Lia`. Keyword argument allows for the use of
            this function with `uhf` and `pbc` modules.

        Returns
        -------
        integral : numpy.ndarray
            Main integral.
        """

        # Get the integral intermediates
        if Lia is None:
            Lia = self.integrals.Lia
        naux, nov = Lia.shape  # This `nov` is actually self.mpi_size(nov)

        # Initialise the integral
        dim = 3 if self.report_quadrature_error else 1
        integral = np.zeros((dim, naux, nov))
        integral[:] += Lia

        # Calculate the integral for each point
        for i, (point, weight) in enumerate(zip(*quad)):
            f = self.d / (self.d**2 + point**2)
            q = np.dot(Lia * f[None], Lia.T) * 4.0  # aux^2 o v
            q = mpi_helper.allreduce(q)
            tmp = np.linalg.inv(np.eye(naux) + q) - np.eye(naux)
            del q

            contrib = weight * np.dot(tmp, Lia * f[None]) * (2 / (np.pi))

            integral[0] += contrib
            if i % 2 == 0 and self.report_quadrature_error:
                integral[1] += 2 * contrib
            if i % 4 == 0 and self.report_quadrature_error:
                integral[2] += 4 * contrib

        return integral

    def gen_ClenCur_quad_semiinf(self):
        """Generate quadrature points and weights for Clenshaw-Curtis quadrature over semiinfinite
        range (0 to +inf)
        """
        tvals = [(np.pi * j / (self.gw.npoints + 1)) for j in range(1, self.gw.npoints + 1)]
        points = np.asarray([1.0 / (np.tan(t / 2) ** 2) for t in tvals])
        jsums = [
            sum(
                [np.sin(j * t) * (1 - np.cos(j * np.pi)) / j for j in range(1, self.gw.npoints + 1)]
            )
            for t in tvals
        ]
        weights = np.asarray(
            [
                1.0 * (4 * np.sin(t) / ((self.gw.npoints + 1) * (1 - np.cos(t)) ** 2)) * s
                for (t, s) in zip(tvals, jsums)
            ]
        )
        return points, weights

    def gen_gausslag_quad_semiinf(self):
        """Generate quadrature points and weights for Gauss-Laguerre quadrature over an ``(0,
        +inf)``.

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
        """Estimate the quadrature error for Clenshaw-Curtis quadrature.

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
