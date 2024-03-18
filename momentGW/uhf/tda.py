"""
Construct TDA moments with unrestricted references.
"""

import numpy as np

from momentGW import logging, mpi_helper, util
from momentGW.tda import dTDA as RdTDA


class dTDA(RdTDA):
    """
    Compute the self-energy moments using dTDA and numerical
    integration with unrestricted references.

    Parameters
    ----------
    gw : BaseUGW
        GW object.
    nmom_max : int
        Maximum moment number to calculate.
    integrals : UIntegrals
        Integrals object.
    mo_energy : dict, optional
        Molecular orbital energies for each spin. Keys are "g" and "w"
        for the Green's function and screened Coulomb interaction,
        respectively. If `None`, use `gw.mo_energy` for both. Default
        value is `None`.
    mo_occ : dict, optional
        Molecular orbital occupancies for each spin. Keys are "g" and
        "w" for the Green's function and screened Coulomb interaction,
        respectively. If `None`, use `gw.mo_occ` for both. Default
        value is `None`.
    """

    @logging.with_timer("Density-density moments")
    @logging.with_status("Constructing density-density moments")
    def build_dd_moments(self):
        """Build the moments of the density-density response.

        Returns
        -------
        moments : tuple of numpy.ndarray
            Moments of the density-density response for each spin
            channel.
        """

        a0, a1 = self.mpi_slice(self.nov[0])
        b0, b1 = self.mpi_slice(self.nov[1])
        moments = np.zeros((self.nmom_max + 1, self.naux, (a1 - a0) + (b1 - b0)))

        # Construct energy differences
        d = np.concatenate(
            [
                util.build_1h1p_energies(self.mo_energy_w[0], self.mo_occ_w[0]).ravel()[a0:a1],
                util.build_1h1p_energies(self.mo_energy_w[1], self.mo_occ_w[1]).ravel()[b0:b1],
            ]
        )

        # Get the zeroth order moment
        moments[0] = np.concatenate([self.integrals[0].Lia, self.integrals[1].Lia], axis=1)

        # Get the higher order moments
        for i in range(1, self.nmom_max + 1):
            moments[i] = moments[i - 1] * d[None]
            tmp = np.dot(moments[i - 1], moments[0].T)
            tmp = mpi_helper.allreduce(tmp)
            moments[i] += np.dot(tmp, moments[0])
            del tmp

        return moments

    build_dd_moments_exact = build_dd_moments

    @logging.with_timer("Self-energy moments")
    @logging.with_status("Constructing self-energy moments")
    def build_se_moments(self, moments_dd):
        """Build the moments of the self-energy via convolution.

        Parameters
        ----------
        moments_dd : numpy.ndarray
            Moments of the density-density response for each spin
            channel.

        Returns
        -------
        moments_occ : numpy.ndarray
            Moments of the occupied self-energy for each spin channel.
        moments_vir : numpy.ndarray
            Moments of the virtual self-energy for each spin channel.
        """

        # Setup dependent on diagonal SE
        a0, a1 = self.mpi_slice(self.mo_energy_g[0].size)
        b0, b1 = self.mpi_slice(self.mo_energy_g[1].size)
        if self.gw.diagonal_se:
            eta = [
                np.zeros((a1 - a0, self.nmo[0])),
                np.zeros((b1 - b0, self.nmo[1])),
            ]
            pq = p = q = "p"
        else:
            eta = [
                np.zeros((a1 - a0, self.nmo[0], self.nmo[0])),
                np.zeros((b1 - b0, self.nmo[1], self.nmo[1])),
            ]
            pq, p, q = "pq", "p", "q"

        Lia = np.concatenate([self.integrals[0].Lia, self.integrals[1].Lia], axis=1)

        # Initialise output moments
        moments_occ = [
            np.zeros((self.nmom_max + 1, self.nmo[0], self.nmo[0])),
            np.zeros((self.nmom_max + 1, self.nmo[1], self.nmo[1])),
        ]
        moments_vir = [
            np.zeros((self.nmom_max + 1, self.nmo[0], self.nmo[0])),
            np.zeros((self.nmom_max + 1, self.nmo[1], self.nmo[1])),
        ]

        # Get the moments in (aux|aux) and rotate to (mo|mo)
        for n in range(self.nmom_max + 1):
            eta_aux = np.dot(moments_dd[n], Lia.T)
            eta_aux = mpi_helper.allreduce(eta_aux)

            for x in range(a1 - a0):
                Lp = self.integrals[0].Lpx[:, :, x]
                eta[0][x] = util.einsum(f"P{p},Q{q},PQ->{pq}", Lp, Lp, eta_aux)

            for x in range(b1 - b0):
                Lp = self.integrals[1].Lpx[:, :, x]
                eta[1][x] = util.einsum(f"P{p},Q{q},PQ->{pq}", Lp, Lp, eta_aux)

            # Construct the self-energy moments for this order only
            # to save memory
            moments_occ_n, moments_vir_n = self.convolve(
                eta[0][:, None],
                eta_orders=[n],
                mo_energy_g=self.mo_energy_g[0],
                mo_occ_g=self.mo_occ_g[0],
            )
            moments_occ[0] += moments_occ_n
            moments_vir[0] += moments_vir_n
            moments_occ_n, moments_vir_n = self.convolve(
                eta[1][:, None],
                eta_orders=[n],
                mo_energy_g=self.mo_energy_g[1],
                mo_occ_g=self.mo_occ_g[1],
            )
            moments_occ[1] += moments_occ_n
            moments_vir[1] += moments_vir_n

        return tuple(moments_occ), tuple(moments_vir)

    @property
    def nov(self):
        """Number of ov states in the screened Coulomb interaction."""
        return (
            np.sum(self.mo_occ_w[0] > 0) * np.sum(self.mo_occ_w[0] == 0),
            np.sum(self.mo_occ_w[1] > 0) * np.sum(self.mo_occ_w[1] == 0),
        )
