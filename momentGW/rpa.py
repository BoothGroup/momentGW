"""
Construct RPA moments.
"""

import numpy as np
import scipy.optimize
import scipy.special
from pyscf import lib
from pyscf.agf2 import mpi_helper


class RPA:
    """
    Compute the self-energy moments using dRPA and numerical integration.

    Parameters
    ----------
    gw : BaseGW
        GW object.
    nmom_max : int
        Maximum moment number to calculate.
    Lpx : numpy.ndarray
        Density-fitted ERI tensor. `p` is in the basis of MOs, `x` is in
        the basis of the Green's function.
    Lia : numpy.ndarray
        Density-fitted ERI tensor for the occupied-virtual slice. `i` and
        `a` are in the basis of the screened Coulomb interaction.
    mo_energy : numpy.ndarray or tuple of numpy.ndarray, optional
        Molecular orbital energies.  If a tuple is passed, the first
        element corresponds to the Green's function basis and the second to
        the screened Coulomb interaction.  Default value is that of
        `gw._scf.mo_energy`.
    mo_occ : numpy.ndarray or tuple of numpy.ndarray, optional
        Molecular orbital occupancies.  If a tuple is passed, the first
        element corresponds to the Green's function basis and the second to
        the screened Coulomb interaction.  Default value is that of
        `gw._scf.mo_occ`.
    """

    def __init__(
        self,
        gw,
        nmom_max,
        Lpx,
        Lia,
        mo_energy=None,
        mo_occ=None,
    ):
        self.gw = gw
        self.nmom_max = nmom_max
        self.Lpx = Lpx
        self.Lia = Lia

        # Get the MO energies for G and W
        if mo_energy is None:
            self.mo_energy_g = self.mo_energy_w = gw._scf.mo_energy
        elif isinstance(mo_energy, tuple):
            self.mo_energy_g, self.mo_energy_w = mo_energy
        else:
            self.mo_energy_g = self.mo_energy_w = mo_energy

        # Get the MO occupancies for G and W
        if mo_occ is None:
            self.mo_occ_g = self.mo_occ_w = gw._scf.mo_occ
        elif isinstance(mo_occ, tuple):
            self.mo_occ_g, self.mo_occ_w = mo_occ
        else:
            self.mo_occ_g = self.mo_occ_w = mo_occ

        # Reshape ERI tensors
        self.Lia = self.Lia.reshape(self.naux, self.nov)
        self.Lpx = self.Lpx.reshape(self.naux, self.nmo, self.mo_energy_g.size)

        # Options and thresholds
        self.report_quadrature_error = True
        self.compress_ov_threshold = 1e-10

    def kernel(self, exact=False):
        """Run the RIRPA to compute moments of the self-energy."""

        lib.logger.info(self.gw, "Constructing RPA moments (nmom_max = %d)", self.nmom_max)
        if mpi_helper.size > 1:
            lib.logger.info(
                self.gw,
                "Slice of W space on proc %d: [%d, %d]",
                mpi_helper.rank,
                *self.mpi_slice(self.nov),
            )
            lib.logger.info(
                self.gw,
                "Slice of G space on proc %d: [%d, %d]",
                mpi_helper.rank,
                *self.mpi_slice(self.mo_energy_g.size),
            )

        self.compress_eris()

        if exact:
            moments_dd = self.build_dd_moments_exact()
        else:
            integral = self.integrate()
            moments_dd = self.build_dd_moments(integral)

        moments_occ, moments_vir = self.build_se_moments(moments_dd)

        return moments_occ, moments_vir

    def compress_eris(self):
        """Compress the ERI tensors."""

        if self.compress_ov_threshold is None or self.compress_ov_threshold < 1e-14:
            return

        cput0 = (lib.logger.process_clock(), lib.logger.perf_counter())
        naux_init = self.naux

        tmp = np.dot(self.Lia, self.Lia.T)
        tmp = mpi_helper.reduce(tmp, root=0)
        if mpi_helper.rank == 0:
            e, v = np.linalg.eigh(tmp)
            mask = np.abs(e) > self.compress_ov_threshold
            rot = v[:, mask]
        else:
            rot = np.zeros((0,))
        del tmp

        rot = mpi_helper.bcast(rot, root=0)

        self.Lia = lib.einsum("L...,LQ->Q...", self.Lia, rot)
        self.Lpx = lib.einsum("L...,LQ->Q...", self.Lpx, rot)

        lib.logger.info(
            self.gw,
            "Compressed ERI auxiliary space from %d to %d",
            naux_init,
            self.naux,
        )
        lib.logger.timer(self.gw, "compressing ERIs", *cput0)

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
        diag_eri[p0:p1] = lib.einsum("np,np->p", self.Lia, self.Lia)
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

    def build_dd_moments(self, integral):
        """Build the moments of the density-density response."""

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
        Liad = self.Lia * d[None]
        Liadinv = self.Lia / d[None]

        # Construct (A-B)^{-1}
        u = np.dot(Liadinv, self.Lia.T) * 4.0  # aux^2 o v
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
            tmp = np.dot(moments[i - 2], self.Lia.T)  # aux^2 o v
            tmp = mpi_helper.allreduce(tmp)
            moments[i] += np.dot(tmp, Liad) * 4.0  # aux^2 o v
            del tmp
            cput1 = lib.logger.timer(self.gw, "moment %d" % i, *cput1)

        return moments

    def build_dd_moments_exact(self):
        """Build the exact moments of the density-density response."""

        cput0 = (lib.logger.process_clock(), lib.logger.perf_counter())
        lib.logger.info(self.gw, "Building exact density-density moments")
        lib.logger.debug(self.gw, "Memory usage: %.2f GB", self._memory_usage())

        import sys

        sys.argv.append("--silent")
        from vayesta.rpa import ssRPA

        rpa = ssRPA(self.gw._scf)
        rpa.kernel()

        rot = np.concatenate([self.Lia, self.Lia], axis=-1)

        moments = rpa.gen_moms(self.nmom_max)
        moments = lib.einsum("nij,Pi->nPj", moments, rot)

        return moments[:, :, : self.nov]

    def build_se_moments(self, moments_dd):
        """Build the moments of the self-energy via convolution."""

        cput0 = (lib.logger.process_clock(), lib.logger.perf_counter())
        lib.logger.info(self.gw, "Building self-energy moments")
        lib.logger.debug(self.gw, "Memory usage: %.2f GB", self._memory_usage())

        p0, p1 = self.mpi_slice(self.nov)
        q0, q1 = self.mpi_slice(self.mo_energy_g.size)

        # Setup dependent on diagonal SE
        if self.gw.diagonal_se:
            pq = p = q = "p"
            eta = np.zeros((q1 - q0, self.nmom_max + 1, self.nmo))
            fproc = lambda x: np.diag(x)
        else:
            pq, p, q = "pq", "p", "q"
            eta = np.zeros((q1 - q0, self.nmom_max + 1, self.nmo, self.nmo))
            fproc = lambda x: x

        # Get the moments in (aux|aux) and rotate to (mo|mo)
        for n in range(self.nmom_max + 1):
            eta_aux = np.dot(moments_dd[n], self.Lia.T)  # aux^2 o v
            eta_aux = mpi_helper.allreduce(eta_aux)
            for x in range(q1 - q0):
                Lp = self.Lpx[:, :, x]
                eta[x, n] = lib.einsum(f"P{p},Q{q},PQ->{pq}", Lp, Lp, eta_aux) * 2.0
        cput1 = lib.logger.timer(self.gw, "rotating DD moments", *cput0)

        # Construct the self-energy moments
        moments_occ = np.zeros((self.nmom_max + 1, self.nmo, self.nmo))
        moments_vir = np.zeros((self.nmom_max + 1, self.nmo, self.nmo))
        moms = np.arange(self.nmom_max + 1)
        for n in moms:
            fp = scipy.special.binom(n, moms)
            fh = fp * (-1) ** moms
            if np.any(self.mo_occ_g[q0:q1] > 0):
                eo = np.power.outer(self.mo_energy_g[self.mo_occ_g > 0][q0:q1], n - moms)
                to = lib.einsum(f"t,kt,kt{pq}->{pq}", fh, eo, eta[self.mo_occ_g[q0:q1] > 0])
                moments_occ[n] += fproc(to)
            if np.any(self.mo_occ_g[q0:q1] == 0):
                ev = np.power.outer(self.mo_energy_g[self.mo_occ_g == 0][q0:q1], n - moms)
                tv = lib.einsum(f"t,ct,ct{pq}->{pq}", fp, ev, eta[self.mo_occ_g[q0:q1] == 0])
                moments_vir[n] += fproc(tv)
        moments_occ = mpi_helper.allreduce(moments_occ)
        moments_vir = mpi_helper.allreduce(moments_vir)
        moments_occ = 0.5 * (moments_occ + moments_occ.swapaxes(1, 2))
        moments_vir = 0.5 * (moments_vir + moments_vir.swapaxes(1, 2))
        cput1 = lib.logger.timer(self.gw, "constructing SE moments", *cput1)

        return moments_occ, moments_vir

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

        Liad = self.Lia * d[None]
        integral = np.zeros((self.naux, self.mpi_size(self.nov)))

        for point, weight in zip(*quad):
            expval = np.exp(-point * d)
            lhs = np.dot(Liad * expval[None], self.Lia.T)  # aux^2 o v
            lhs = mpi_helper.allreduce(lhs)
            rhs = self.Lia * expval[None]  # aux o v
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
        Liad = self.Lia * d[None]
        integral = np.zeros((dim, self.naux, p1 - p0))

        for i, (point, weight) in enumerate(zip(*quad)):
            f = 1.0 / (d**2 + point**2)
            q = np.dot(self.Lia * f[None], Liad.T) * 4.0  # aux^2 o v
            q = mpi_helper.allreduce(q)
            tmp = np.linalg.inv(np.eye(self.naux) + q) - np.eye(self.naux)

            contrib = np.linalg.multi_dot((q, tmp, self.Lia))  # aux^2 o v
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

    def _memory_usage(self):
        """Return the current memory usage in GB."""
        return lib.current_memory()[0] / 1e3

    @property
    def nmo(self):
        """Number of MOs."""
        return self.gw.nmo

    @property
    def naux(self):
        """Number of auxiliaries."""
        assert self.Lpx.shape[0] == self.Lia.shape[0]
        return self.Lpx.shape[0]

    @property
    def nov(self):
        """Number of ov states in W."""
        return np.sum(self.mo_occ_w > 0) * np.sum(self.mo_occ_w == 0)

    def mpi_slice(self, n):
        """
        Return the start and end index for the current process for total
        size `n`.
        """
        return list(mpi_helper.prange(0, n, n))[0]

    def mpi_size(self, n):
        """
        Return the number of states in the current process for total size
        `n`.
        """
        p0, p1 = self.mpi_slice(n)
        return p1 - p0
