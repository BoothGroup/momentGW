"""
Construct RPA moments.
"""

import numpy as np
import scipy.optimize
from pyscf import lib
from pyscf.agf2 import mpi_helper

from momentGW import TDA


class RPA(TDA):
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
    mo_energy : numpy.ndarray or tuple of numpy.ndarray, optional
        Molecular orbital energies. If a tuple is passed, the first
        element corresponds to the Green's function basis and the second to
        the screened Coulomb interaction. Default value is that of
        `gw._scf.mo_energy`.
    mo_occ : numpy.ndarray or tuple of numpy.ndarray, optional
        Molecular orbital occupancies. If a tuple is passed, the first
        element corresponds to the Green's function basis and the second to
        the screened Coulomb interaction. Default value is that of
        `gw._scf.mo_occ`.
    """

    def integrate(self):
        """Optimise the quadrature and perform the integration.

        Returns
        -------
        integral : numpy.ndarray
            Integral array, including the offset part.
        """

        cput0 = (lib.logger.process_clock(), lib.logger.perf_counter())
        lib.logger.info(self.gw, "Performing integration")
        lib.logger.debug(self.gw, "Memory usage: %.2f GB", self._memory_usage())
        p0, p1 = self.mpi_slice(self.nov)

        # Construct energy differences
        d_full = lib.direct_sum(
            "a-i->ia",
            self.mo_energy_w[self.mo_occ_w == 0],
            self.mo_energy_w[self.mo_occ_w > 0],
        ).ravel()
        d = d_full[p0:p1]

        # Calculate diagonal part of ERI
        diag_eri = np.zeros((self.nov,))
        diag_eri[p0:p1] = lib.einsum("np,np->p", self.integrals.Lia, self.integrals.Lia)
        diag_eri = mpi_helper.allreduce(diag_eri)

        # Get the offset integral quadrature
        quad = self.optimise_offset_quad(d_full, diag_eri)
        cput1 = lib.logger.timer(self.gw, "optimising offset quadrature", *cput0)

        # Perform the offset integral
        offset = self.eval_offset_integral(quad, d)
        cput1 = lib.logger.timer(self.gw, "performing offset integral", *cput1)

        # Get the main integral quadrature
        quad = self.optimise_main_quad(d_full, diag_eri)
        cput1 = lib.logger.timer(self.gw, "optimising main quadrature", *cput1)

        # Perform the main integral
        integral = self.eval_main_integral(quad, d)
        cput1 = lib.logger.timer(self.gw, "performing main integral", *cput1)

        # Report quadrature error
        if self.report_quadrature_error:
            a = np.sum((integral[0] - integral[2]) ** 2)
            b = np.sum((integral[0] - integral[1]) ** 2)
            a, b = mpi_helper.allreduce(np.array([a, b]))
            a, b = a**0.5, b**0.5
            err = self.estimate_error_clencur(a, b)
            lib.logger.debug(self.gw, "One-quarter quadrature error: %s", a)
            lib.logger.debug(self.gw, "One-half quadrature error: %s", b)
            lib.logger.debug(self.gw, "Error estimate: %s", err)

        return integral[0] + offset

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

        cput0 = (lib.logger.process_clock(), lib.logger.perf_counter())
        lib.logger.info(self.gw, "Building density-density moments")
        lib.logger.debug(self.gw, "Memory usage: %.2f GB", self._memory_usage())

        p0, p1 = self.mpi_slice(self.nov)
        moments = np.zeros((self.nmom_max + 1, self.naux, p1 - p0))

        # Construct energy differences
        d_full = lib.direct_sum(
            "a-i->ia",
            self.mo_energy_w[self.mo_occ_w == 0],
            self.mo_energy_w[self.mo_occ_w > 0],
        ).ravel()
        d = d_full[p0:p1]

        # Calculate (L|ia) D_{ia} and (L|ia) D_{ia}^{-1} intermediates
        Liad = self.integrals.Lia * d[None]
        Liadinv = self.integrals.Lia / d[None]

        # Construct (A-B)^{-1}
        u = np.dot(Liadinv, self.integrals.Lia.T) * 4.0  # aux^2 o v
        u = mpi_helper.allreduce(u)
        u = np.linalg.inv(np.eye(self.naux) + u)
        cput1 = lib.logger.timer(self.gw, "constructing (A-B)^{-1}", *cput0)

        # Get the zeroth order moment
        moments[0] = integral / d[None]
        tmp = np.linalg.multi_dot((integral, Liadinv.T, u))  # aux^2 o v
        tmp = mpi_helper.allreduce(tmp)
        moments[0] -= np.dot(tmp, Liadinv) * 4.0  # aux^2 o v
        del u, tmp
        cput1 = lib.logger.timer(self.gw, "zeroth moment", *cput1)

        # Get the first order moment
        moments[1] = Liad
        cput1 = lib.logger.timer(self.gw, "first moment", *cput1)

        # Get the higher order moments
        for i in range(2, self.nmom_max + 1):
            moments[i] = moments[i - 2] * d[None] ** 2
            tmp = np.dot(moments[i - 2], self.integrals.Lia.T)  # aux^2 o v
            tmp = mpi_helper.allreduce(tmp)
            moments[i] += np.dot(tmp, Liad) * 4.0  # aux^2 o v
            del tmp
            cput1 = lib.logger.timer(self.gw, "moment %d" % i, *cput1)

        return moments

    def build_dd_moments_exact(self):
        """Build the exact moments of the density-density response.

        Returns
        -------
        moments : numpy.ndarray
            Moments of the density-density response.
        """

        cput0 = (lib.logger.process_clock(), lib.logger.perf_counter())
        lib.logger.info(self.gw, "Building exact density-density moments")
        lib.logger.debug(self.gw, "Memory usage: %.2f GB", self._memory_usage())

        import sys

        sys.argv.append("--silent")
        from vayesta.rpa import ssRPA

        rpa = ssRPA(self.gw._scf)
        rpa.kernel()

        rot = np.concatenate([self.integrals.Lia, self.integrals.Lia], axis=-1)

        moments = rpa.gen_moms(self.nmom_max)
        moments = lib.einsum("nij,Pi->nPj", moments, rot)

        return moments[:, :, : self.nov]

    # --- Numerical integration functions:

    @staticmethod
    def rescale_quad(bare_quad, a):
        """Rescale quadrature for grid space `a`."""
        return bare_quad[0] * a, bare_quad[1] * a

    def optimise_main_quad(self, d, diag_eri):
        """
        Optimise the grid spacing of Clenshaw-Curtis quadrature for the
        main integral.

        Parameters
        ----------
        d : numpy.ndarray
            Array of orbital energy differences.
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

    def optimise_offset_quad(self, d, diag_eri):
        """
        Optimise the grid spacing of Clenshaw-Curtis quadrature for the
        main integral.

        Parameters
        ----------
        d : numpy.ndarray
            Array of orbital energy differences.
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

        exact = 0.5 * np.dot(1.0 / d, d * diag_eri)

        integrand = lambda quad: self.eval_diag_offset_integral(quad, d, diag_eri)
        quad = self.get_optimal_quad(bare_quad, integrand, exact)

        return quad

    def get_optimal_quad(self, bare_quad, integrand, exact):
        """Get the optimal quadrature.

        Parameters
        ----------
        bare_quad : tuple
            The quadrature points and weights.
        integrand : function
            The integrand function.
        exact : float
            The exact value of the integral.

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
        lib.logger.debug(
            self.gw,
            "Used minimisation to optimise quadrature grid: a = %.2e  penalty = %.2e",
            solve,
            res.fun,
        )

        return self.rescale_quad(bare_quad, solve)

    def eval_diag_offset_integral(self, quad, d, diag_eri):
        """Evaluate the diagonal of the offset integral.

        Parameters
        ----------
        quad : tuple
            The quadrature points and weights.
        d : numpy.ndarray
            Array of orbital energy differences.
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
            integral += res * weight  # aux^2 o v

        return integral

    def eval_diag_main_integral(self, quad, d, diag_eri):
        """Evaluate the diagonal of the main integral.

        Parameters
        ----------
        quad : tuple
            The quadrature points and weights.
        d : numpy.ndarray
            Array of orbital energy differences.
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

    def eval_offset_integral(self, quad, d):
        """Evaluate the offset integral.

        Parameters
        ----------
        quad : tuple
            The quadrature points and weights.
        d : numpy.ndarray
            Array of orbital energy differences.

        Returns
        -------
        integral : numpy.ndarray
            Offset integral.
        """

        Liad = self.integrals.Lia * d[None]
        integral = np.zeros((self.naux, self.mpi_size(self.nov)))

        for point, weight in zip(*quad):
            expval = np.exp(-point * d)
            lhs = np.dot(Liad * expval[None], self.integrals.Lia.T)  # aux^2 o v
            lhs = mpi_helper.allreduce(lhs)
            rhs = self.integrals.Lia * expval[None]  # aux o v
            res = np.dot(lhs, rhs)
            integral += res * weight  # aux^2 o v

        integral *= 4.0
        integral += Liad

        return integral

    def eval_main_integral(self, quad, d):
        """Evaluate the main integral.

        Parameters
        ----------
        quad : tuple
            The quadrature points and weights.
        d : numpy.ndarray
            Array of orbital energy differences.

        Returns
        -------
        integral : numpy.ndarray
            Offset integral.
        """

        p0, p1 = self.mpi_slice(self.nov)
        dim = 3 if self.report_quadrature_error else 1
        Liad = self.integrals.Lia * d[None]
        integral = np.zeros((dim, self.naux, p1 - p0))

        for i, (point, weight) in enumerate(zip(*quad)):
            f = 1.0 / (d**2 + point**2)
            q = np.dot(self.integrals.Lia * f[None], Liad.T) * 4.0  # aux^2 o v
            q = mpi_helper.allreduce(q)
            tmp = np.linalg.inv(np.eye(self.naux) + q) - np.eye(self.naux)

            contrib = np.linalg.multi_dot((q, tmp, self.integrals.Lia))  # aux^2 o v
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
            lib.logger.warning(
                self.gw,
                "Nested quadrature error estimation gives %d real roots. "
                "Taking smallest positive root." % len(real_roots),
            )
        else:
            lib.logger.debug(
                self.gw,
                "Nested quadrature error estimation gives %d real roots." % len(real_roots),
            )

        # Check if there is a root between 0 and 1
        if not np.any(np.logical_and(real_roots > 0, real_roots < 1)):
            lib.logger.critical(
                self.gw, "Nested quadrature error estimation gives no root between 0 and 1."
            )
            return np.nan
        else:
            root = np.min(real_roots[np.logical_and(real_roots > 0, real_roots < 1)])

        # Calculate the error
        error = i2 / (1.0 + root**-2)

        return error
