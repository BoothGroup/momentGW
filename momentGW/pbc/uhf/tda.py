"""
Construct TDA moments with periodic boundary conditions and unrestricted
references.
"""

import functools

import numpy as np

from momentGW import logging, mpi_helper, util
from momentGW.pbc.tda import dTDA as KdTDA
from momentGW.uhf.tda import dTDA as MolUdTDA


class dTDA(KdTDA, MolUdTDA):
    """
    Compute the self-energy moments using dTDA with periodic boundary
    conditions and unrestricted references.

    Parameters
    ----------
    gw : BaseKUGW
        GW object.
    nmom_max : int
        Maximum moment number to calculate.
    integrals : KUIntegrals
        Integrals object.
    mo_energy : dict, optional
        Molecular orbital energies at each k-point for each spin channel.
        Keys are "g" and "w" for the Green's function and screened
        Coulomb interaction, respectively. If `None`, use `gw.mo_energy`
        for both. Default value is `None`.
    mo_occ : dict, optional
        Molecular orbital occupancies at each k-point for each spin
        channel. Keys are "g" and "w" for the Green's function and
        screened Coulomb interaction, respectively. If `None`, use
        `gw.mo_occ` for both. Default value is `None`.
    """

    @logging.with_timer("Density-density moments")
    @logging.with_status("Constructing density-density moments")
    def build_dd_moments(self):
        """Build the moments of the density-density response.

        Returns
        -------
        moments : numpy.ndarray
            Moments of the density-density response at each k-point
            for each spin channel.
        """

        # Initialise the moments
        kpts = self.kpts
        naux = self.naux
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

                tmp = np.zeros((naux[q], naux[q]), dtype=complex)
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

        return moments

    def kernel(self, exact=False):
        """
        Run the polarizability calculation to compute moments of the
        self-energy.

        Parameters
        ----------
        exact : bool, optional
            Has no effect and is only present for compatibility with
            `dRPA`. Default value is `False`.

        Returns
        -------
        moments_occ : numpy.ndarray
            Moments of the occupied self-energy at each k-point for each
            spin channel.
        moments_vir : numpy.ndarray
            Moments of the virtual self-energy at each k-point for each
            spin channel.
        """
        return super().kernel(exact=exact)

    @logging.with_timer("Moment convolution")
    @logging.with_status("Convoluting moments")
    def convolve(self, eta, mo_energy_g=None, mo_occ_g=None):
        """
        Handle the convolution of the moments of the Green's function
        and screened Coulomb interaction.

        Parameters
        ----------
        eta : numpy.ndarray
            Moments of the density-density response partly transformed
            into moments of the screened Coulomb interaction, at each
            k-point for each spin channel.
        mo_energy_g : numpy.ndarray, optional
            Energies of the Green's function at each k-point for each
            spin channel. If `None`, use `self.mo_energy_g`. Default
            value is `None`.
        mo_occ_g : numpy.ndarray, optional
            Occupancies of the Green's function at each k-point for each
            spin channel. If `None`, use `self.mo_occ_g`. Default value
            is `None`.

        Returns
        -------
        moments_occ : numpy.ndarray
            Moments of the occupied self-energy at each k-point for each
            spin channel.
        moments_vir : numpy.ndarray
            Moments of the virtual self-energy at each k-point for each
            spin channel.
        """
        return super().convolve(
            eta,
            mo_energy_g=mo_energy_g,
            mo_occ_g=mo_occ_g,
        )

    @logging.with_timer("Self-energy moments")
    @logging.with_status("Constructing self-energy moments")
    def build_se_moments(self, moments_dd):
        """Build the moments of the self-energy via convolution.

        Parameters
        ----------
        moments_dd : numpy.ndarray
            Moments of the density-density response at each k-point.

        Returns
        -------
        moments_occ : numpy.ndarray
            Moments of the occupied self-energy at each k-point for each
            spin channel.
        moments_vir : numpy.ndarray
            Moments of the virtual self-energy at each k-point for each
            spin channel.
        """

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
                            eta[s, kp, q][x, n] += util.einsum(subscript, Lp, Lp.conj(), eta_aux)

        # Construct the self-energy moments
        moments_occ = [None, None]
        moments_vir = [None, None]
        moments_occ[0], moments_vir[0] = self.convolve(
            eta[0], mo_energy_g=self.mo_energy_g[0], mo_occ_g=self.mo_occ_g[0]
        )
        moments_occ[1], moments_vir[1] = self.convolve(
            eta[1], mo_energy_g=self.mo_energy_g[1], mo_occ_g=self.mo_occ_g[1]
        )

        return tuple(moments_occ), tuple(moments_vir)

    @functools.cached_property
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


class TDAx(dTDA):
    """
    Compute the self-energy moments using TDA (with exchange) with
    periodic boundary conditions and unrestricted references.

    Parameters
    ----------
    gw : BaseKUGW
        GW object.
    nmom_max : int
        Maximum moment number to calculate.
    integrals : KUIntegrals
        Integrals object.
    mo_energy : dict, optional
        Molecular orbital energies at each k-point for each spin channel.
        Keys are "g" and "w" for the Green's function and screened
        Coulomb interaction, respectively. If `None`, use `gw.mo_energy`
        for both. Default value is `None`.
    mo_occ : dict, optional
        Molecular orbital occupancies at each k-point for each spin
        channel. Keys are "g" and "w" for the Green's function and
        screened Coulomb interaction, respectively. If `None`, use
        `gw.mo_occ` for both. Default value is `None`.
    """

    @logging.with_timer("Self-energy moments")
    @logging.with_status("Constructing self-energy moments")
    def build_se_moments(self, moments_dd):
        """Build the moments of the self-energy via convolution.

        Parameters
        ----------
        moments_dd : numpy.ndarray
            Moments of the density-density response at each k-point.

        Returns
        -------
        moments_occ : numpy.ndarray
            Moments of the occupied self-energy at each k-point for each
            spin channel.
        moments_vir : numpy.ndarray
            Moments of the virtual self-energy at each k-point for each
            spin channel.
        """

        # Get the sizes
        nocc = self.integrals.nocc  # noqa # FIXME
        nvir = self.integrals.nvir  # noqa # FIXME
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
                            eta[s, kp, q][x, n] += util.einsum(subscript, Lp, Lp.conj(), eta_aux)

        # Construct the self-energy moments
        moments_occ = [None, None]
        moments_vir = [None, None]
        moments_occ[0], moments_vir[0] = self.convolve(
            eta[0], mo_energy_g=self.mo_energy_g[0], mo_occ_g=self.mo_occ_g[0]
        )
        moments_occ[1], moments_vir[1] = self.convolve(
            eta[1], mo_energy_g=self.mo_energy_g[1], mo_occ_g=self.mo_occ_g[1]
        )

        return tuple(moments_occ), tuple(moments_vir)
