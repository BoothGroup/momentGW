"""
Construct RPA moments with periodic boundary conditions.
"""

import numpy as np
from pyscf import lib

from momentGW import mpi_helper, util
from momentGW.pbc.tda import dTDA
from momentGW.rpa import dRPA as MoldRPA

# TODO: Check lack of Lai in the integrals


class dRPA(dTDA, MoldRPA):
    """
    Compute the self-energy moments using dRPA and numerical integration
    with periodic boundary conditions.

    Parameters
    ----------
    gw : BaseKGW
        GW object.
    nmom_max : int
        Maximum moment number to calculate.
    Lpx : numpy.ndarray
        Density-fitted ERI tensor, where the first two indices
        enumerate the k-points (as a dict), the third index is the auxiliary
        basis function index, and the fourth and fifth indices are
        the MO and Green's function orbital indices, respectively.
    integrals : KIntegrals
        Density-fitted integrals.
    mo_energy : dict, optional
        Molecular orbital energies. Keys are "g" and "w" for the Green's
        function and screened Coulomb interaction, respectively.
        If `None`, use `gw.mo_energy` for both. Default value is `None`.
    mo_occ : dict, optional
        Molecular orbital occupancies. Keys are "g" and "w" for the
        Green's function and screened Coulomb interaction, respectively.
        If `None`, use `gw.mo_occ` for both. Default value is `None`.
    """

    def _build_d(self):
        """Construct the energy differences matrix.
        """

        d = np.zeros((self.nkpts, self.nkpts), dtype=object)

        for q in self.kpts.loop(1):
            for ki in self.kpts.loop(1, mpi=True):
                ka = self.kpts.member(self.kpts.wrap_around(self.kpts[q] + self.kpts[ki]))
                d[q, ka] = util.build_1h1p_energies(
                    (self.mo_energy_w[ki], self.mo_energy_w[ka]),
                    (self.mo_occ_w[ki], self.mo_occ_w[ka]),
                ).ravel()

        return d

    def _build_diag_eri(self):
        """Construct the diagonal of the ERIs for each k-point.
        """

        diag_eri = np.zeros((self.nkpts, self.nkpts), dtype=object)

        for q in self.kpts.loop(1):
            for ki in self.kpts.loop(1, mpi=True):
                kb = self.kpts.member(self.kpts.wrap_around(self.kpts[q] + self.kpts[ki]))
                diag_eri[q, kb] = np.sum(np.abs(self.integrals.Lia[ki, kb]) ** 2, axis=0) / self.nkpts

        return diag_eri

    def _build_Liad(self, Lia, d):
        """Construct the Liad array.
        """

        Liad = np.zeros((self.nkpts, self.nkpts), dtype=object)

        for q in self.kpts.loop(1):
            for ki in self.kpts.loop(1, mpi=True):
                kb = self.kpts.member(self.kpts.wrap_around(self.kpts[q] + self.kpts[ki]))
                Liad[q, kb] = Lia[ki, kb] * d[q, kb]

        return Liad

    def _build_Liadinv(self, Lia, d):
        """Construct the Liadinv array.
        """

        Liadinv = np.zeros((self.nkpts, self.nkpts), dtype=object)

        for q in self.kpts.loop(1):
            for ki in self.kpts.loop(1, mpi=True):
                kb = self.kpts.member(self.kpts.wrap_around(self.kpts[q] + self.kpts[ki]))
                Liadinv[q, kb] = Lia[ki, kb] / d[q, kb]

        return Liadinv

    def integrate(self):
        """
        Optimise the quadrature and perform the integration for a given
        set of k points.

        Returns
        -------
        integral : numpy.ndarray
            Integral array, including the offset part.
        """

        cput0 = (lib.logger.process_clock(), lib.logger.perf_counter())
        lib.logger.info(self.gw, "Performing integration")
        lib.logger.debug(self.gw, "Memory usage: %.2f GB", self._memory_usage())

        # Construct the energy differences
        d = self._build_d()

        # Calculate diagonal part of ERIs
        diag_eri = self._build_diag_eri()

        # Get the offset integral quadrature
        quad = self.optimise_offset_quad(d, diag_eri)
        cput1 = lib.logger.timer(self.gw, "optimising offset quadrature", *cput0)

        # Perform the offset integral
        offset = self.eval_offset_integral(quad, d)
        cput1 = lib.logger.timer(self.gw, "performing offset integral", *cput1)

        # Get the main integral quadrature
        quad = self.optimise_main_quad(d, diag_eri)
        cput1 = lib.logger.timer(self.gw, "optimising main quadrature", *cput1)

        # Perform the main integral
        integral = self.eval_main_integral(quad, d)
        cput1 = lib.logger.timer(self.gw, "performing main integral", *cput1)

        # Report quadrature error
        if self.report_quadrature_error:
            a = 0.0
            b = 0.0
            for q in self.kpts.loop(1):
                for ka in self.kpts.loop(1, mpi=True):
                    a += np.sum((integral[0, q, ka] - integral[2, q, ka]) ** 2)
                    b += np.sum((integral[0, q, ka] - integral[1, q, ka]) ** 2)
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

        Variables
        ----------
        diag_eri : numpy.ndarray
            Diagonal of the ERIs for each k-point.
        d : numpy.ndarray
            Array of orbital energy differences for each k-point.
        Lia : dict of numpy.ndarray
            Dict. with keys that are pairs of k-point indices (Nkpt, Nkpt) with an array of form (aux, W occ, W vir) at
            this k-point pair. The 1st Nkpt is defined by the difference between k-points and the second index's kpoint.
        Liad : dict of numpy.ndarray
            Dict. with keys that are pairs of k-point indices (Nkpt, Nkpt) with an array of form (aux, W occ, W vir)
            at this k-point pair. Liad is formed from the multiplication of the Lia array and the orbital energy
             differences.
        Liadinv : dict of numpy.ndarray
            Dict. with keys that are pairs of k-point indices (Nkpt, Nkpt) with an array of form (aux, W occ, W vir)
            at this k-point pair. Liadinv is formed from the division of the Lia array and the orbital energy
             differences.
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

        kpts = self.kpts
        Lia = self.integrals.Lia
        moments = np.zeros((self.nkpts, self.nkpts, self.nmom_max + 1), dtype=object)

        # Construct the energy differences
        d = self._build_d()

        # Calculate (L|ia) D_{ia} and (L|ia) D_{ia}^{-1} intermediates
        Liad = self._build_Liad(Lia, d)
        Liadinv = self._build_Liadinv(Lia, d)

        for q in kpts.loop(1):
            # Get the zeroth order moment
            tmp = np.zeros((self.naux[q], self.naux[q]), dtype=complex)
            inter = 0.0
            for kj in kpts.loop(1, mpi=True):
                kb = kpts.member(kpts.wrap_around(kpts[q] + kpts[kj]))
                tmp += np.dot(Liadinv[q, kb], self.integrals.Lia[kj, kb].conj().T)
                inter += np.dot(integral[q, kb], Liadinv[q, kb].T.conj())
            tmp = mpi_helper.allreduce(tmp)
            inter = mpi_helper.allreduce(inter)
            tmp *= 2.0
            u = np.linalg.inv(np.eye(tmp.shape[0]) * self.nkpts / 2 + tmp)

            rest = np.dot(inter, u) * self.nkpts / 2
            for ki in kpts.loop(1, mpi=True):
                ka = kpts.member(kpts.wrap_around(kpts[q] + kpts[ki]))
                moments[q, ka, 0] = integral[q, ka] / d[q, ka] * self.nkpts / 2
                moments[q, ka, 0] -= np.dot(rest, Liadinv[q, ka]) * 2
        cput1 = lib.logger.timer(self.gw, "zeroth moment", *cput0)

        # Get the first order moment
        moments[:, :, 1] = Liad / self.nkpts
        cput1 = lib.logger.timer(self.gw, "first moment", *cput1)

        # Get the higher order moments
        for i in range(2, self.nmom_max + 1):
            for q in kpts.loop(1):
                tmp = 0.0
                for ka in kpts.loop(1, mpi=True):
                    kb = kpts.member(kpts.wrap_around(kpts[q] + kpts[ka]))
                    moments[q, kb, i] = moments[q, kb, i - 2] * d[q, kb] ** 2
                    tmp += np.dot(moments[q, kb, i - 2], self.integrals.Lia[ka, kb].conj().T)
                tmp = mpi_helper.allreduce(tmp)
                tmp /= self.nkpts
                tmp *= 2
                for ki in kpts.loop(1, mpi=True):
                    ka = kpts.member(kpts.wrap_around(kpts[q] + kpts[ki]))
                    moments[q, ka, i] += np.dot(tmp, Liad[q, ka]) * 2
            cput1 = lib.logger.timer(self.gw, "moment %d" % i, *cput1)

        return moments

    def optimise_offset_quad(self, d, diag_eri):
        """
        Optimise the grid spacing of Gauss-Laguerre quadrature for the
        offset integral.

        Variables
        ----------
        d : numpy.ndarray
            Array of orbital energy differences for each k-point.
        diag_eri : numpy.ndarray
            Diagonal of the ERIs for each k-point.

        Returns
        -------
        points : numpy.ndarray
            The quadrature points.
        weights : numpy.ndarray
            The quadrature weights.
        """

        bare_quad = self.gen_gausslag_quad_semiinf()
        exact = 0.0
        for q in self.kpts.loop(1):
            for ki in self.kpts.loop(1, mpi=True):
                ka = self.kpts.member(self.kpts.wrap_around(self.kpts[q] + self.kpts[ki]))
                exact += np.dot(1.0 / d[q, ka], d[q, ka] * diag_eri[q, ka])
        exact = mpi_helper.allreduce(exact)
        integrand = lambda quad: self.eval_diag_offset_integral(quad, d, diag_eri)
        quad = self.get_optimal_quad(bare_quad, integrand, exact)
        return quad

    def eval_diag_offset_integral(self, quad, d, diag_eri):
        """Evaluate the diagonal of the offset integral.

        Parameters
        ----------
        quad : tuple
            The quadrature points and weights.

        Variables
        ----------
        d : numpy.ndarray
            Array of orbital energy differences for each k-point.
        diag_eri : numpy.ndarray
            Diagonal of the ERIs for each k-point.

        Returns
        -------
        integral : numpy.ndarray
            Offset integral.
        """

        integral = 0.0

        for point, weight in zip(*quad):
            for q in self.kpts.loop(1):
                for ki in self.kpts.loop(1, mpi=True):
                    ka = self.kpts.member(self.kpts.wrap_around(self.kpts[q] + self.kpts[ki]))
                    tmp = d[q, ka] * diag_eri[q, ka]
                    expval = np.exp(-2 * point * d[q, ka])
                    res = np.dot(expval, tmp)
                    integral += 2 * res * weight
        integral = mpi_helper.allreduce(integral)
        return integral

    def eval_offset_integral(self, quad, d, Lia=None):
        """Evaluate the offset integral.

        Parameters
        ----------
        quad : tuple
            The quadrature points and weights.

        Variables
        ----------
        d : numpy.ndarray
            Array of orbital energy differences for each k-point.
        Lia : dict of numpy.ndarray
            Dict. with keys that are pairs of k-point indices (Nkpt, Nkpt) with an array of form (aux, W occ, W vir) at
            this k-point pair. The 1st Nkpt is defined by the difference between k-points and the second index's kpoint.
        Liad : dict of numpy.ndarray
            Dict. with keys that are pairs of k-point indices (Nkpt, Nkpt) with an array of form (aux, W occ, W vir)
            at this k-point pair. See "build_dd_moments" for more details.

        Returns
        -------
        integral : numpy.ndarray
            Offset integral.
        """

        if Lia is None:
            Lia = self.integrals.Lia

        Liad = self._build_Liad(Lia, d)
        integrals = 2 * Liad / (self.nkpts**2)

        kpts = self.kpts

        for point, weight in zip(*quad):
            for q in kpts.loop(1):
                lhs = 0.0
                for ka in kpts.loop(1, mpi=True):
                    kb = kpts.member(kpts.wrap_around(kpts[q] + kpts[ka]))
                    expval = np.exp(-point * d[q, kb])
                    lhs += np.dot(Liad[q, kb] * expval[None], Lia[ka, kb].T.conj())
                lhs = mpi_helper.allreduce(lhs)
                lhs /= self.nkpts
                lhs *= 2

                for ka in kpts.loop(1, mpi=True):
                    kb = kpts.member(kpts.wrap_around(kpts[q] + kpts[ka]))
                    rhs = self.integrals.Lia[ka, kb] * np.exp(-point * d[q, kb])
                    rhs /= self.nkpts**2
                    res = np.dot(lhs, rhs)
                    integrals[q, kb] += res * weight * 4

        return integrals

    def optimise_main_quad(self, d, diag_eri):
        """
        Optimise the grid spacing of Clenshaw-Curtis quadrature for the
        main integral.

        Variables
        ----------
        d : numpy.ndarray
            Array of orbital energy differences for each k-point.
        diag_eri : numpy.ndarray
            Diagonal of the ERIs for each k-point.

        Returns
        -------
        points : numpy.ndarray
            The quadrature points.
        weights : numpy.ndarray
            The quadrature weights.
        """

        bare_quad = self.gen_clencur_quad_inf(even=True)

        exact = 0.0
        d_sq = np.zeros((self.nkpts, self.nkpts), dtype=object)
        d_eri = np.zeros((self.nkpts, self.nkpts), dtype=object)
        d_sq_eri = np.zeros((self.nkpts, self.nkpts), dtype=object)
        for q in self.kpts.loop(1):
            for kj in self.kpts.loop(1, mpi=True):
                kb = self.kpts.member(self.kpts.wrap_around(self.kpts[q] + self.kpts[kj]))
                exact += np.sum((d[q, kb] * (d[q, kb] + diag_eri[q, kb])) ** 0.5)
                exact -= 0.5 * np.dot(1.0 / d[q, kb], d[q, kb] * diag_eri[q, kb])
                exact -= np.sum(d[q, kb])
                d_sq[q, kb] = d[q, kb] ** 2
                d_eri[q, kb] = d[q, kb] * diag_eri[q, kb]
                d_sq_eri[q, kb] = d[q, kb] * (d[q, kb] + diag_eri[q, kb])
        exact = mpi_helper.allreduce(exact)
        integrand = lambda quad: self.eval_diag_main_integral(quad, d, d_sq, d_eri, d_sq_eri)
        quad = self.get_optimal_quad(bare_quad, integrand, exact)
        return quad

    def eval_diag_main_integral(self, quad, d, d_sq, d_eri, d_sq_eri):
        """Evaluate the diagonal of the main integral.

        Parameters
        ----------
        quad : tuple
            The quadrature points and weights.
        d_sq : numpy.ndarray
            Array of orbital energy differences squared for each k-point. See
            "optimise_main_quad" for more details.
        d_eri : numpy.ndarray
            Array of orbital energy differences times the diagonal of the ERIs for each
            k-point. See "optimise_main_quad" for more details.
        d_sq_eri : numpy.ndarray
            Array of orbital energy differences times the diagonal of the ERIs plus the orbital
            energy differences for each k-point. See "optimise_main_quad" for more details.

        Variables
        ----------
        d : numpy.ndarray
            Array of orbital energy differences for each k-point.
        diag_eri : numpy.ndarray
            Diagonal of the ERIs for each k-point.

        Returns
        -------
        integral : numpy.ndarray
            Main integral.
        """

        integral = 0.0

        def diag_contrib(x, freq):
            integral = x / (x + freq**2)
            integral /= np.pi
            return integral

        for point, weight in zip(*quad):
            contrib = 0.0
            for q in self.kpts.loop(1):
                for kj in self.kpts.loop(1, mpi=True):
                    kb = self.kpts.member(self.kpts.wrap_around(self.kpts[q] + self.kpts[kj]))
                    contrib_int = diag_contrib(d_sq_eri[q, kb], point)
                    contrib_int -= diag_contrib(d_sq[q, kb], point)
                    contrib += np.abs(np.sum(contrib_int))

                    f_sq = 1.0 / (d[q, kb] ** 2 + point**2) ** 2
                    contrib -= np.abs(point**2 * np.dot(f_sq, d_eri[q, kb]) / np.pi)
            integral += weight * contrib
        integral = mpi_helper.allreduce(integral)
        return integral

    def eval_main_integral(self, quad, d, Lia=None):
        """Evaluate the main integral.

        Parameters
        ----------
        quad : tuple
            The quadrature points and weights.

        Variables
        ----------
        d : numpy.ndarray
            Array of orbital energy differences for each k-point.
        Lia : dict of numpy.ndarray
            Dict. with keys that are pairs of k-point indices (Nkpt, Nkpt) with an array of form (aux, W occ, W vir) at
            this k-point pair. The 1st Nkpt is defined by the difference between k-points and the second index's kpoint.
        Liad : dict of numpy.ndarray
            Dict. with keys that are pairs of k-point indices (Nkpt, Nkpt) with an array of form (aux, W occ, W vir)
            at this k-point pair. See "build_dd_moments" for more details.

        Returns
        -------
        integral : numpy.ndarray
            Offset integral.
        """

        if Lia is None:
            Lia = self.integrals.Lia

        Liad = self._build_Liad(Lia, d)
        dim = 3 if self.report_quadrature_error else 1
        integral = np.zeros((dim, self.nkpts, self.nkpts), dtype=object)
        kpts = self.kpts
        for i, (point, weight) in enumerate(zip(*quad)):
            contrib = np.zeros_like(d, dtype=object)

            for q in kpts.loop(1):
                f = np.zeros((self.nkpts), dtype=object)
                qz = 0.0
                for ki in kpts.loop(1, mpi=True):
                    kj = kpts.member(kpts.wrap_around(kpts[q] + kpts[ki]))
                    f[kj] = 1.0 / (d[q, kj] ** 2 + point**2)
                    pre = (Lia[ki, kj] * f[kj]) * (4 / self.nkpts)
                    qz += np.dot(pre, Liad[q, kj].T.conj())
                qz = mpi_helper.allreduce(qz)

                tmp = np.linalg.inv(np.eye(self.naux[q]) + qz) - np.eye(self.naux[q])
                inner = np.dot(qz, tmp)

                for ka in kpts.loop(1, mpi=True):
                    kb = kpts.member(kpts.wrap_around(kpts[q] + kpts[ka]))
                    contrib[q, kb] = (
                        2 * np.dot(inner, Lia[ka, kb]) / (self.nkpts**2)
                    )
                    value = weight * (contrib[q, kb] * f[kb] * (point**2 / np.pi))

                    integral[0, q, kb] += value
                    if i % 2 == 0 and self.report_quadrature_error:
                        integral[1, q, kb] += 2 * value
                    if i % 4 == 0 and self.report_quadrature_error:
                        integral[2, q, kb] += 4 * value

        return integral
