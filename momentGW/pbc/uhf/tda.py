"""
Construct TDA moments with periodic boundary conditions and unrestricted
references.
"""

import numpy as np
from pyscf import lib

from momentGW import mpi_helper, util
from momentGW.pbc.tda import dTDA as KdTDA
from momentGW.uhf.tda import dTDA as MolUdTDA


class dTDA(KdTDA, MolUdTDA):
    """
    Compute the self-energy moments using dTDA and numerical
    integration with periodic boundary conditions and unrestricted
    references.

    Parameters
    ----------
    gw : BaseKUGW
        GW object.
    nmom_max : int
        Maximum moment number to calculate.
    integrals : KUIntegrals
        Integrals object.
    mo_energy : tuple of (numpy.ndarray or tuple of numpy.ndarray), optional
        Molecular orbital energies for each spin channel. If either
        element is a tuple, the first element corresponds to the Green's
        function basis and the second to the screened Coulomb
        interaction. Default value is that of `gw.mo_energy`.
    mo_occ : tuple of (numpy.ndarray or tuple of numpy.ndarray), optional
        Molecular orbital occupancies for each spin channel. If either
        element is a tuple, the first element corresponds to the Green's
        function basis and the second to the screened Coulomb
        interaction. Default value is that of `gw.mo_occ`.
    """

    def build_dd_moments(self):
        """Build the moments of the density-density response.

        Returns
        -------
        moments : numpy.ndarray
            Moments of the density-density response for each k-point
            for each spin channel.
        """

        cput0 = (lib.logger.process_clock(), lib.logger.perf_counter())
        lib.logger.info(self.gw, "Building density-density moments")
        lib.logger.debug(self.gw, "Memory usage: %.2f GB", self._memory_usage())

        kpts = self.kpts
        moments = np.zeros((self.nkpts, self.nkpts, self.nmom_max + 1), dtype=object)

        # Get the zeroth order moment
        for q in kpts.loop(1):
            for kj in kpts.loop(1, mpi=True):
                kb = kpts.member(kpts.wrap_around(kpts[q] + kpts[kj]))
                moments[q, kb, 0] += (
                    np.concatenate(
                        [
                            self.integrals[0].Lia[kj, kb],
                            self.integrals[1].Lia[kj, kb],
                        ],
                        axis=1,
                    )
                    / self.nkpts
                )
        cput1 = lib.logger.timer(self.gw, "zeroth moment", *cput0)

        # Get the higher order moments
        for i in range(1, self.nmom_max + 1):
            for q in kpts.loop(1):
                for kj in kpts.loop(1, mpi=True):
                    kb = kpts.member(kpts.wrap_around(kpts[q] + kpts[kj]))

                    d = np.concatenate(
                        [
                            util.build_1h1p_energies(
                                (self.mo_energy_w[0][kj], self.mo_energy_w[0][kb]),
                                (self.mo_occ_w[0][kj], self.mo_occ_w[0][kb]),
                            ).ravel(),
                            util.build_1h1p_energies(
                                (self.mo_energy_w[1][kj], self.mo_energy_w[1][kb]),
                                (self.mo_occ_w[1][kj], self.mo_occ_w[1][kb]),
                            ).ravel(),
                        ]
                    )
                    moments[q, kb, i] += moments[q, kb, i - 1] * d[None]

                tmp = np.zeros((self.naux[q], self.naux[q]), dtype=complex)
                for ki in kpts.loop(1, mpi=True):
                    ka = kpts.member(kpts.wrap_around(kpts[q] + kpts[ki]))

                    Lia = np.concatenate(
                        [
                            self.integrals[0].Lia[ki, ka],
                            self.integrals[1].Lia[ki, ka],
                        ],
                        axis=1,
                    )

                    tmp += np.dot(moments[q, ka, i - 1], Lia.T.conj())

                tmp = mpi_helper.allreduce(tmp)
                tmp /= self.nkpts

                for kj in kpts.loop(1, mpi=True):
                    kb = kpts.member(kpts.wrap_around(kpts[q] + kpts[kj]))

                    Lai = np.concatenate(
                        [
                            self.integrals[0].Lai[kj, kb],
                            self.integrals[1].Lai[kj, kb],
                        ],
                        axis=1,
                    )

                    moments[q, kb, i] += np.dot(tmp, Lai.conj())

            cput1 = lib.logger.timer(self.gw, "moment %d" % i, *cput1)

        return moments

    build_dd_moments_exact = build_dd_moments

    def build_se_moments(self, moments_dd):
        """Build the moments of the self-energy via convolution.

        Parameters
        ----------
        moments_dd : numpy.ndarray
            Moments of the density-density response for each k-point.

        Returns
        -------
        moments_occ : numpy.ndarray
            Moments of the occupied self-energy for each k-point.
        moments_vir : numpy.ndarray
            Moments of the virtual self-energy for each k-point.
        """

        cput0 = (lib.logger.process_clock(), lib.logger.perf_counter())
        lib.logger.info(self.gw, "Building self-energy moments")
        lib.logger.debug(self.gw, "Memory usage: %.2f GB", self._memory_usage())

        kpts = self.kpts

        # Setup dependent on diagonal SE
        if self.gw.diagonal_se:
            pqchar = pchar = qchar = "p"
            eta_shape = lambda s, k: (self.mo_energy_g[s][k].size, self.nmom_max + 1, self.nmo[s])
        else:
            pqchar, pchar, qchar = "pq", "p", "q"
            eta_shape = lambda s, k: (
                self.mo_energy_g[s][k].size,
                self.nmom_max + 1,
                self.nmo[s],
                self.nmo[s],
            )
        eta = np.zeros((2, self.nkpts, self.nkpts), dtype=object)

        # Get the moments in (aux|aux) and rotate to (mo|mo)
        for n in range(self.nmom_max + 1):
            for q in kpts.loop(1):
                eta_aux = 0
                for kj in kpts.loop(1, mpi=True):
                    kb = kpts.member(kpts.wrap_around(kpts[q] + kpts[kj]))
                    Lia = np.concatenate(
                        [
                            self.integrals[0].Lia[kj, kb],
                            self.integrals[1].Lia[kj, kb],
                        ],
                        axis=1,
                    )
                    eta_aux += np.dot(moments_dd[q, kb, n], Lia.T.conj())

                eta_aux = mpi_helper.allreduce(eta_aux)
                eta_aux /= self.nkpts

                for kp in kpts.loop(1, mpi=True):
                    kx = kpts.member(kpts.wrap_around(kpts[kp] - kpts[q]))

                    for s in range(2):
                        if not isinstance(eta[s, kp, q], np.ndarray):
                            eta[s, kp, q] = np.zeros(eta_shape(s, kx), dtype=eta_aux.dtype)

                        for x in range(self.mo_energy_g[s][kx].size):
                            Lp = self.integrals[s].Lpx[kp, kx][:, :, x]
                            subscript = f"P{pchar},Q{qchar},PQ->{pqchar}"
                            eta[s, kp, q][x, n] += lib.einsum(subscript, Lp, Lp.conj(), eta_aux)

        cput1 = lib.logger.timer(self.gw, "rotating DD moments", *cput0)

        # Construct the self-energy moments
        moments_occ = [None, None]
        moments_vir = [None, None]
        moments_occ[0], moments_vir[0] = self.convolve(
            eta[0], mo_energy_g=self.mo_energy_g[0], mo_occ_g=self.mo_occ_g[0]
        )
        moments_occ[1], moments_vir[1] = self.convolve(
            eta[1], mo_energy_g=self.mo_energy_g[1], mo_occ_g=self.mo_occ_g[1]
        )
        cput1 = lib.logger.timer(self.gw, "constructing SE moments", *cput1)

        return tuple(moments_occ), tuple(moments_vir)

    @property
    def nov(self):
        """Number of ov states in the screened Coulomb interaction."""
        return (
            np.multiply.outer(
                [np.sum(occ > 0) for occ in self.mo_occ_w[0]],
                [np.sum(occ == 0) for occ in self.mo_occ_w[0]],
            ),
            np.multiply.outer(
                [np.sum(occ > 0) for occ in self.mo_occ_w[1]],
                [np.sum(occ == 0) for occ in self.mo_occ_w[1]],
            ),
        )
