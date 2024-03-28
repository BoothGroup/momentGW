"""
Construct GF2 self-energy moments.
"""

import numpy as np

from momentGW import logging, mpi_helper, util
from momentGW.tda import BaseSE


class GF2(BaseSE):
    """
    Compute the GF2 self-energy moments.

    This follows the same structure as the dTDA and dRPA classes,
    allowing conversion of the GW solvers to GF2 solvers.

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
    """

    def __init__(self, *args, non_dyson=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.non_dyson = non_dyson

        # Check the MO energies and occupancies
        if not (
            np.allclose(self.mo_energy_g, self.mo_energy_w)
            and np.allclose(self.mo_occ_g, self.mo_occ_w)
        ):
            raise ValueError(
                "MO energies for Green's function and screened Coulomb "
                f"interaction must be the same for {self.__class__.__name__}."
            )

    def kernel(self):
        """
        Run the polarizability calculation to compute moments of the
        self-energy.

        Returns
        -------
        moments_occ : numpy.ndarray
            Moments of the occupied self-energy.
        moments_vir : numpy.ndarray
            Moments of the virtual self-energy.
        """
        return self.build_se_moments()

    @logging.with_timer("Self-energy moments")
    @logging.with_status("Constructing self-energy moments")
    def build_se_moments(self):
        """Build the moments of the self-energy.

        Returns
        -------
        moments_occ : numpy.ndarray
            Moments of the occupied self-energy.
        moments_vir : numpy.ndarray
            Moments of the virtual self-energy.
        """

        # Setup dependent on diagonal SE
        if self.gw.diagonal_se:
            pq = p = q = "p"
            fproc = lambda x: np.diag(x)
        else:
            pq, p, q = "pq", "p", "q"
            fproc = lambda x: x

        # Get the physical space mask
        if self.non_dyson:
            po = slice(None, np.sum(self.mo_occ_g > 0))
            pv = slice(np.sum(self.mo_occ_g > 0), None)
        else:
            po = slice(None)
            pv = slice(None)

        # Initialise output moments
        moments_occ = np.zeros((self.nmom_max + 1, self.nmo, self.nmo))
        moments_vir = np.zeros((self.nmom_max + 1, self.nmo, self.nmo))

        # Get the energies
        e_i = self.mo_energy_g[self.mo_occ_g > 0]
        e_a = self.mo_energy_g[self.mo_occ_g == 0]

        # Get the integrals
        Lia = self.integrals.Lia.reshape(-1, e_i.size, e_a.size)
        Lai = Lia.swapaxes(1, 2)

        # Get the occupied moments
        for i in range(*self.mpi_slice(e_i.size)):
            pija = util.einsum("Pp,Pja->pja", self.integrals.Lpx[:, po, i], Lia)
            pjia = util.einsum("Ppj,Pa->pja", self.integrals.Lpx[:, po, : e_i.size], Lia[:, i])
            pjia = 2 * pija - pjia
            e_ija = e_i[i] + e_i[:, None] - e_a[None, :]

            # Loop over orders
            for n in range(self.nmom_max + 1):
                moments_occ[n, po, po] += fproc(util.einsum(f"{p}ja,{q}ja->{pq}", pija, pjia))
                if n != self.nmom_max:
                    pija = util.einsum("pja,ja->pja", pija, e_ija)

        # Get the virtual moments
        for a in range(*self.mpi_slice(e_a.size)):
            pabi = util.einsum("Pp,Pbi->pbi", self.integrals.Lpx[:, pv, e_i.size + a], Lai)
            pbai = util.einsum("Ppb,Pi->pbi", self.integrals.Lpx[:, pv, e_i.size :], Lai[:, a])
            pbai = 2 * pabi - pbai
            e_abi = e_a[a] + e_a[:, None] - e_i[None, :]

            # Loop over orders
            for n in range(self.nmom_max + 1):
                moments_vir[n, pv, pv] += fproc(util.einsum(f"{p}bi,{q}bi->{pq}", pabi, pbai))
                if n != self.nmom_max:
                    pabi = util.einsum("pbi,bi->pbi", pabi, e_abi)

        moments_occ = mpi_helper.allreduce(moments_occ)
        moments_vir = mpi_helper.allreduce(moments_vir)

        return moments_occ, moments_vir
