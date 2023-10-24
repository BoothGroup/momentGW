"""
Construct RPA moments with periodic boundary conditions.
"""

import numpy as np
import scipy.optimize
import scipy.special
from pyscf import lib

from momentGW import mpi_helper, util
from momentGW.pbc.tda import dTDA
from momentGW.rpa import dRPA as MoldRPA

# TODO: Check lack of Lai in the integrals


class dRPA(MoldRPA, dTDA):
    """
    Compute the self-energy moments using dTDA and numerical integration
    with periodic boundary conditions.

    Parameters
    ----------
    gw : BaseKGW
        GW object.
    nmom_max : int
        Maximum moment number to calculate.
    mo_energy : dict, optional
        Molecular orbital energies. Keys are "g" and "w" for the Green's
        function and screened Coulomb interaction, respectively.
        If `None`, use `gw.mo_energy` for both. Default value is `None`.
    mo_occ : dict, optional
        Molecular orbital occupancies. Keys are "g" and "w" for the
        Green's function and screened Coulomb interaction, respectively.
        If `None`, use `gw.mo_occ` for both. Default value is `None`.
    """

    def integrate(self):
        """Optimise the quadrature and perform the integration for a
        given set of k points.

        Parameters
        ----------
        q : int
            Current index of k point loop

        Returns
        -------
        integral : numpy.ndarray
            Integral array, including the offset part.
        """

        kpts = self.kpts

        cput0 = (lib.logger.process_clock(), lib.logger.perf_counter())
        lib.logger.info(self.gw, "Performing integration")
        lib.logger.debug(self.gw, "Memory usage: %.2f GB", self._memory_usage())

        offset = np.zeros((self.nkpts, self.nkpts), dtype=object)
        main = np.zeros((self.nkpts, self.nkpts), dtype=object)
        d = np.zeros((self.nkpts, self.nkpts), dtype=object)
        diag_eri = np.zeros((self.nkpts, self.nkpts), dtype=object)
        Liad = np.zeros((self.nkpts, self.nkpts), dtype=object)
        for q in kpts.loop(1):
            for kj in kpts.loop(1, mpi=True):
                kb = kpts.member(kpts.wrap_around(kpts[q] + kpts[kj]))
                d[q, kb] = util.build_1h1p_energies(
                    (self.mo_energy_w[kj], self.mo_energy_w[kb]),
                    (self.mo_occ_w[kj], self.mo_occ_w[kb]),
                ).ravel()
                diag_eri[q, kb] = (
                    lib.einsum(
                        "np,np->p", self.integrals.Lia[kj, kb].conj(), self.integrals.Lia[kj, kb]
                    )
                    + lib.einsum(
                        "np,np->p", self.integrals.Lia[kj, kb].conj(), self.integrals.Lia[kj, kb]
                    )
                ) / (2 * self.nkpts)
                # diag_eri[q,kb] = ((lib.einsum("np,np->p", self.integrals.Lia[kj, kb].conj(), self.integrals.Lia[kj, kb])
                #                    + lib.einsum("np,np->p", self.integrals.Lia[kj, kb].conj(), self.integrals.Lai[kj, kb]))
                #                   /(2*self.nkpts))
                Liad[q, kb] = self.integrals.Lia[kj, kb] * d[q, kb]

        # Get the offset integral quadrature
        quad = self.optimise_offset_quad(d, diag_eri)
        cput1 = lib.logger.timer(self.gw, "optimising offset quadrature", *cput0)

        # Perform the offset integral
        self.eval_offset_integral(quad, d, Liad, offset)
        cput1 = lib.logger.timer(self.gw, "performing offset integral", *cput1)

        # Get the main integral quadrature
        quad = self.optimise_main_quad(d, diag_eri)
        cput1 = lib.logger.timer(self.gw, "optimising main quadrature", *cput1)

        # Perform the main integral
        self.eval_main_integral(quad, d, Liad, main)
        cput1 = lib.logger.timer(self.gw, "performing main integral", *cput1)
        # TODO: Implement quadrature error reporting
        # Report quadrature error
        # if self.report_quadrature_error:
        #     a = np.sum((integral[0] - integral[2]) ** 2)
        #     b = np.sum((integral[0] - integral[1]) ** 2)
        #     a, b = mpi_helper.allreduce(np.array([a, b]))
        #     a, b = a ** 0.5, b ** 0.5
        #     err = self.estimate_error_clencur(a, b)
        #     lib.logger.debug(self.gw, "One-quarter quadrature error: %s", a)
        #     lib.logger.debug(self.gw, "One-half quadrature error: %s", b)
        #     lib.logger.debug(self.gw, "Error estimate: %s", err)

        return main + offset

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

        kpts = self.kpts
        if integral is None:
            integral = self.integrate()

        moments = np.zeros((self.nkpts, self.nkpts, self.nmom_max + 1), dtype=object)

        d = np.zeros((self.nkpts, self.nkpts), dtype=object)
        Liad = np.zeros((self.nkpts, self.nkpts), dtype=object)
        Laidinv = np.zeros((self.nkpts, self.nkpts), dtype=object)
        Liadinv = np.zeros((self.nkpts, self.nkpts), dtype=object)

        for q in kpts.loop(1):
            for kj in kpts.loop(1, mpi=True):
                kb = kpts.member(kpts.wrap_around(kpts[q] + kpts[kj]))
                d[q, kb] = util.build_1h1p_energies(
                    (self.mo_energy_w[kj], self.mo_energy_w[kb]),
                    (self.mo_occ_w[kj], self.mo_occ_w[kb]),
                ).ravel()
                Liad[q, kb] = self.integrals.Lia[kj, kb] * d[q, kb]
                Laidinv[q, kb] = (
                    self.integrals.Lia[kj, kb] / d[q, kb]
                )  # +self.integrals.Lai[kj, kb]/d[q, kb])#(self.integrals.Lia[kj, kb]/d[q, kb]+self.integrals.Lai[kj, kb]/d[q, kb])
                Liadinv[q, kb] = self.integrals.Lia[kj, kb] / d[q, kb]
        for q in kpts.loop(1):
            tmp = np.zeros((self.naux[q], self.naux[q]), dtype=complex)
            for ki in kpts.loop(1, mpi=True):
                ka = kpts.member(kpts.wrap_around(kpts[q] + kpts[ki]))
                tmp += np.dot(Laidinv[q, ka], self.integrals.Lia[ki, ka].conj().T)
            tmp = mpi_helper.allreduce(tmp)
            tmp *= 2.0
            u = np.linalg.inv(np.eye(tmp.shape[0]) * self.nkpts / 2 + tmp)

            # Get the zeroth order moment
            inter = 0.0
            for kj in kpts.loop(1, mpi=True):
                kb = kpts.member(kpts.wrap_around(kpts[q] + kpts[kj]))
                inter += np.dot(integral[q, kb], Liadinv[q, kb].T.conj())

            rest = np.dot(inter, u) * self.nkpts / 2
            for ki in kpts.loop(1, mpi=True):
                ka = kpts.member(kpts.wrap_around(kpts[q] + kpts[ki]))
                moments[q, ka, 0] = integral[q, ka] / d[q, ka] * self.nkpts / 2
                moments[q, ka, 0] -= np.dot(rest, Laidinv[q, ka]) * 2

        # Get the first order moment
        moments[:, :, 1] = Liad / self.nkpts

        # Get the higher order moments
        for i in range(2, self.nmom_max + 1):
            for q in kpts.loop(1):
                tmp = 0.0
                for kj in kpts.loop(1, mpi=True):
                    kb = kpts.member(kpts.wrap_around(kpts[q] + kpts[kj]))
                    moments[q, kb, i] = moments[q, kb, i - 2] * d[q, kb] ** 2  # * 2 /self.nkpts
                    tmp += (
                        np.dot(moments[q, kb, i - 2], self.integrals.Lia[kj, kb].conj().T)
                        * 2
                        / self.nkpts
                    )  # aux^2 o v
                tmp = mpi_helper.allreduce(tmp)
                for ki in kpts.loop(1, mpi=True):
                    ka = kpts.member(kpts.wrap_around(kpts[q] + kpts[ki]))
                    moments[q, ka, i] += np.dot(tmp, Liad[q, ka]) * 2  # aux^2 o v

        # cput1 = lib.logger.timer(self.gw, "moment %d" % i, *cput1)

        return moments

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
        q : int
            Current index of k point loop

        Returns
        -------
        points : numpy.ndarray
            The quadrature points.
        weights : numpy.ndarrayf
            The quadrature weights.
        """

        bare_quad = self.gen_gausslag_quad_semiinf()
        exact = 0.0
        for q in self.kpts.loop(1):
            for ki in self.kpts.loop(1, mpi=True):
                exact += np.dot(1.0 / d[q, ki], d[q, ki] * diag_eri[q, ki])
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
            Array of orbital energy differences.
        diag_eri : numpy.ndarray
            Diagonal of the ERIs.
        q : int
            Current index of k point loop

        Returns
        -------
        integral : numpy.ndarray
            Offset integral.
        """

        # TODO check this: is this still right, does it need a factor 4.0?

        integral = 0.0

        for point, weight in zip(*quad):
            for q in self.kpts.loop(1):
                # res = np.zeros_like(d, dtype=object)
                for ki in self.kpts.loop(1, mpi=True):
                    ka = self.kpts.member(self.kpts.wrap_around(self.kpts[q] + self.kpts[ki]))
                    tmp = d[q, ka] * diag_eri[q, ka]
                    expval = np.exp(-2 * point * d[q, ka])
                    res = np.dot(expval, tmp)
                    integral += 2 * res * weight
        return integral

    def eval_offset_integral(self, quad, d, Liad, integrals, Lia=None):
        """Evaluate the offset integral.

        Parameters
        ----------
        quad : tuple
            The quadrature points and weights.
        d : numpy.ndarray
            Array of orbital energy differences.
        Lia : numpy.ndarray
            The (aux, W occ, W vir) integral array. If `None`, use
            `self.integrals.Lia`. Keyword argument allows for the use of
            this function with `uhf` and `pbc` modules.
        q : int
            Current index of k point loop

        Returns
        -------
        integral : numpy.ndarray
            Offset integral.
        """

        kpts = self.kpts

        if Lia is None:
            Lia = self.integrals.Lia

        for point, weight in zip(*quad):
            rhs = np.zeros_like(integrals)
            for q in kpts.loop(1):
                lhs = 0.0
                for kj in kpts.loop(1, mpi=True):
                    kb = kpts.member(kpts.wrap_around(kpts[q] + kpts[kj]))
                    lhs += np.dot(Liad[q, kb] * np.exp(-point * d[q, kb]), Lia[kj, kb].T.conj())
                lhs /= self.nkpts

                for kj in kpts.loop(1, mpi=True):
                    kb = kpts.member(kpts.wrap_around(kpts[q] + kpts[kj]))
                    rhs[q, kb] = Lia[kj, kb] * np.exp(-point * d[q, kb]) + self.integrals.Lia[
                        kj, kb
                    ] * np.exp(
                        -point * d[q, kb]
                    )  # aux o v#(Lia[kj,kb]*np.exp(-point * d[q,kb]) + self.integrals.Lai[kj,kb]*np.exp(-point * d[q,kb]))/(2/self.nkpts) # aux o v
                    rhs[q, kb] /= self.nkpts**2
                    rhs[q, kb] = np.dot(lhs, rhs[q, kb])
                    integrals[q, kb] += rhs[q, kb] * weight

            del rhs, lhs

        for q in kpts.loop(1):
            for kj in kpts.loop(1, mpi=True):
                kb = kpts.member(kpts.wrap_around(kpts[q] + kpts[kj]))
                integrals[q, kb] *= 4
                integrals[q, kb] += 2 * Liad[q, kb] / (self.nkpts**2)  # Lai????

        return integrals

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

        exact = 0.0
        tmp = np.zeros_like(d, dtype=object)
        for q in self.kpts.loop(1):
            for kj in self.kpts.loop(1, mpi=True):
                kb = self.kpts.member(self.kpts.wrap_around(self.kpts[q] + self.kpts[kj]))
                tmp[q, kb] = d[q, kb] * diag_eri[q, kb]
        for q in self.kpts.loop(1):
            for kj in self.kpts.loop(1, mpi=True):
                kb = self.kpts.member(self.kpts.wrap_around(self.kpts[q] + self.kpts[kj]))
                exact += np.sum((d[q, kb] * (d[q, kb] + diag_eri[q, kb])) ** 0.5)
                exact -= 0.5 * np.dot(1.0 / d[q, kb], tmp[q, kb])
                exact -= np.sum(d[q, kb])

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
            tmp = np.zeros_like(d, dtype=object)
            f = np.zeros_like(d, dtype=object)
            for q in self.kpts.loop(1):
                for kj in self.kpts.loop(1, mpi=True):
                    kb = self.kpts.member(self.kpts.wrap_around(self.kpts[q] + self.kpts[kj]))
                    tmp[q, kb] = d[kj, kb] * diag_eri[kj, kb]
                    f[q, kb] = 1.0 / (d[q, kb] ** 2 + point**2)
            contrib = 0.0
            for q in self.kpts.loop(1):
                for kj in self.kpts.loop(1, mpi=True):
                    kb = self.kpts.member(self.kpts.wrap_around(self.kpts[q] + self.kpts[kj]))
                    contrib_int = diag_contrib(d[q, kb] * (d[q, kb] + diag_eri[q, kb]), point)
                    contrib_int -= diag_contrib(d[q, kb] ** 2, point)
                    contrib += np.abs(np.sum(contrib_int))

            for q in self.kpts.loop(1):
                for kj in self.kpts.loop(1, mpi=True):
                    kb = self.kpts.member(self.kpts.wrap_around(self.kpts[q] + self.kpts[kj]))
                    contrib -= np.abs(point**2 * np.dot(f[q, kb] ** 2, tmp[q, kb]) / np.pi)
            integral += weight * contrib

        return integral

    def eval_main_integral(self, quad, d, Liad, integral, Lia=None):
        """Evaluate the main integral.

        Parameters
        ----------
        quad : tuple
            The quadrature points and weights.
        d : numpy.ndarray
            Array of orbital energy differences.
        Lia : numpy.ndarray
            The (aux, W occ, W vir) integral array. If `None`, use
            `self.integrals.Lia`. Keyword argument allows for the use of
            this function with `uhf` and `pbc` modules.

        Returns
        -------
        integral : numpy.ndarray
            Offset integral.
        """

        kpts = self.kpts

        if Lia is None:
            Lia = self.integrals.Lia
        for i, (point, weight) in enumerate(zip(*quad)):
            f = np.zeros_like(d, dtype=object)
            contrib = np.zeros_like(d, dtype=object)

            for q in self.kpts.loop(1):
                qz = 0.0
                for kj in self.kpts.loop(1, mpi=True):
                    kb = self.kpts.member(self.kpts.wrap_around(self.kpts[q] + self.kpts[kj]))
                    f[q, kb] = 1.0 / (d[q, kb] ** 2 + point**2)
                    pre = (Lia[kj, kb] * f[q, kb] + self.integrals.Lia[kj, kb] * f[q, kb]) * (
                        2 / self.nkpts
                    )  # (Lia[kj, kb]*f[q,kb] + self.integrals.Lai[kj, kb]*f[q,kb])*(2/self.nkpts)
                    qz += np.dot(pre, Liad[q, kb].T.conj())
                qz = mpi_helper.allreduce(qz)

                tmp = np.linalg.inv(np.eye(self.naux[q]) + qz) - np.eye(self.naux[q])
                inner = np.dot(qz, tmp)

                for ki in self.kpts.loop(1, mpi=True):
                    ka = self.kpts.member(self.kpts.wrap_around(self.kpts[q] + self.kpts[ki]))
                    contrib[q, ka] = 2 * np.dot(inner, Lia[ki, ka]) / (self.nkpts**2)
                    integral[q, ka] += weight * (contrib[q, ka] * f[q, ka] * (point**2 / np.pi))

        return integral

    @property
    def naux(self):
        """Number of auxiliaries."""
        return self.integrals.naux

    @property
    def nov(self):
        """Number of ov states in W."""
        return np.multiply.outer(
            [np.sum(occ > 0) for occ in self.mo_occ_w],
            [np.sum(occ == 0) for occ in self.mo_occ_w],
        )

    @property
    def kpts(self):
        """k-points."""
        return self.gw.kpts

    @property
    def nkpts(self):
        """Number of k-points."""
        return self.gw.nkpts
