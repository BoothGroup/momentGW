"""
Convert the GW solvers to GF2 solvers with unrestricted references.
"""

import numpy as np

from momentGW import util, logging, mpi_helper
from momentGW.gf2 import _GF2 as _RGF2
from momentGW.tda import BaseSE
from momentGW.uhf.base import BaseUGW
from momentGW.uhf.gw import UGW
from momentGW.uhf.evgw import evUGW
from momentGW.uhf.scgw import scUGW
from momentGW.uhf.qsgw import qsUGW
from momentGW.uhf.fsgw import fsUGW


class _GF2(_RGF2):
    """
    Compute the GF2 self-energy moments with unrestricted references.

    This follows the same structure as the dTDA and dRPA classes,
    allowing conversion of the GW solvers to GF2 solvers.

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

    def __init__(self, *args, non_dyson=False, **kwargs):
        BaseSE.__init__(self, *args, **kwargs)
        self.non_dyson = non_dyson

        # Check the MO energies and occupancies
        if not (
            np.allclose(self.mo_energy_g[0], self.mo_energy_w[0])
            and np.allclose(self.mo_energy_g[1], self.mo_energy_w[1])
            and np.allclose(self.mo_occ_g[0], self.mo_occ_w[0])
            and np.allclose(self.mo_occ_g[1], self.mo_occ_w[1])
        ):
            raise ValueError(
                "MO energies for Green's function and screened Coulomb "
                f"interaction must be the same for {self.__class__.__name__}."
            )

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
            po = (
                slice(None, np.sum(self.mo_occ_g[0] > 0)),
                slice(None, np.sum(self.mo_occ_g[1] > 0)),
            )
            pv = (
                slice(np.sum(self.mo_occ_g[0] > 0), None),
                slice(np.sum(self.mo_occ_g[1] > 0), None),
            )
        else:
            po = (slice(None), slice(None))
            pv = (slice(None), slice(None))

        # Initialise output moments
        moments_occ = [
            np.zeros((self.nmom_max + 1, self.nmo[0], self.nmo[0])),
            np.zeros((self.nmom_max + 1, self.nmo[1], self.nmo[1])),
        ]
        moments_vir = [
            np.zeros((self.nmom_max + 1, self.nmo[0], self.nmo[0])),
            np.zeros((self.nmom_max + 1, self.nmo[1], self.nmo[1])),
        ]

        # Get the energies
        e_i = (
            self.mo_energy_g[0][self.mo_occ_g[0] > 0],
            self.mo_energy_g[1][self.mo_occ_g[1] > 0],
        )
        e_a = (
            self.mo_energy_g[0][self.mo_occ_g[0] == 0],
            self.mo_energy_g[1][self.mo_occ_g[1] == 0],
        )

        # Get the integrals
        Lpx = self.integrals.Lpx
        Lia = (
            self.integrals[0].Lia.reshape(self.integrals[0].naux, e_i[0].size, e_a[0].size),
            self.integrals[1].Lia.reshape(self.integrals[1].naux, e_i[1].size, e_a[1].size),
        )
        Lai = (Lia[0].swapaxes(1, 2), Lia[1].swapaxes(1, 2))

        # Get the occupied α moments
        for i in range(*self.mpi_slice(e_i[0].size)):
            pija = util.einsum("Pp,Pja->pja", Lpx[0][:, po[0], i], Lia[0])
            pjia = util.einsum("Ppj,Pa->pja", Lpx[0][:, po[0], : e_i[0].size], Lia[0][:, i])
            piJA = util.einsum("Pp,Pja->pja", Lpx[0][:, po[0], i], Lia[1])
            pija_r = pija - pjia
            piJA_r = piJA.copy()
            e_ija = e_i[0][i] + e_i[0][:, None] - e_a[0][None, :]
            e_iJA = e_i[0][i] + e_i[1][:, None] - e_a[1][None, :]

            # Loop over orders
            for n in range(self.nmom_max + 1):
                moments_occ[0][n, po[0], po[0]] += fproc(util.einsum(f"{p}ja,{q}ja->{pq}", pija, pija_r))
                moments_occ[0][n, po[0], po[0]] += fproc(util.einsum(f"{p}ja,{q}ja->{pq}", piJA, piJA_r))
                if n != self.nmom_max:
                    pija_r = util.einsum("pja,ja->pja", pija_r, e_ija)
                    piJA_r = util.einsum("pja,ja->pja", piJA_r, e_iJA)

        # Get the occupied β moments
        for i in range(*self.mpi_slice(e_i[1].size)):
            pija = util.einsum("Pp,Pja->pja", Lpx[1][:, po[1], i], Lia[1])
            pjia = util.einsum("Ppj,Pa->pja", Lpx[1][:, po[1], : e_i[1].size], Lia[1][:, i])
            piJA = util.einsum("Pp,Pja->pja", Lpx[1][:, po[1], i], Lia[0])
            pija_r = pija - pjia
            piJA_r = piJA.copy()
            e_ija = e_i[1][i] + e_i[1][:, None] - e_a[1][None, :]
            e_iJA = e_i[1][i] + e_i[0][:, None] - e_a[0][None, :]

            # Loop over orders
            for n in range(self.nmom_max + 1):
                moments_occ[1][n, po[1], po[1]] += fproc(util.einsum(f"{p}ja,{q}ja->{pq}", pija, pija_r))
                moments_occ[1][n, po[1], po[1]] += fproc(util.einsum(f"{p}ja,{q}ja->{pq}", piJA, piJA_r))
                if n != self.nmom_max:
                    pija_r = util.einsum("pja,ja->pja", pija_r, e_ija)
                    piJA_r = util.einsum("pja,ja->pja", piJA_r, e_iJA)

        # Get the virtual α moments
        for a in range(*self.mpi_slice(e_a[0].size)):
            pabi = util.einsum("Pp,Pbi->pbi", Lpx[0][:, pv[0], e_i[0].size + a], Lai[0])
            pbai = util.einsum("Ppb,Pi->pbi", Lpx[0][:, pv[0], e_i[0].size :], Lai[0][:, a])
            paBI = util.einsum("Pp,Pbi->pbi", Lpx[0][:, pv[0], e_i[0].size + a], Lai[1])
            pabi_r = pabi - pbai
            paBI_r = paBI.copy()
            e_abi = e_a[0][a] + e_a[0][:, None] - e_i[0][None, :]
            e_aBI = e_a[0][a] + e_a[1][:, None] - e_i[1][None, :]

            # Loop over orders
            for n in range(self.nmom_max + 1):
                moments_vir[0][n, pv[0], pv[0]] += fproc(util.einsum(f"{p}bi,{q}bi->{pq}", pabi, pabi_r))
                moments_vir[0][n, pv[0], pv[0]] += fproc(util.einsum(f"{p}bi,{q}bi->{pq}", paBI, paBI_r))
                if n != self.nmom_max:
                    pabi_r = util.einsum("pbi,bi->pbi", pabi_r, e_abi)
                    paBI_r = util.einsum("pbi,bi->pbi", paBI_r, e_aBI)

        # Get the virtual β moments
        for a in range(*self.mpi_slice(e_a[1].size)):
            pabi = util.einsum("Pp,Pbi->pbi", Lpx[1][:, pv[1], e_i[1].size + a], Lai[1])
            pbai = util.einsum("Ppb,Pi->pbi", Lpx[1][:, pv[1], e_i[1].size :], Lai[1][:, a])
            paBI = util.einsum("Pp,Pbi->pbi", Lpx[1][:, pv[1], e_i[1].size + a], Lai[0])
            pabi_r = pabi - pbai
            paBI_r = paBI.copy()
            e_abi = e_a[1][a] + e_a[1][:, None] - e_i[1][None, :]
            e_aBI = e_a[1][a] + e_a[0][:, None] - e_i[0][None, :]

            # Loop over orders
            for n in range(self.nmom_max + 1):
                moments_vir[1][n, pv[1], pv[1]] += fproc(util.einsum(f"{p}bi,{q}bi->{pq}", pabi, pabi_r))
                moments_vir[1][n, pv[1], pv[1]] += fproc(util.einsum(f"{p}bi,{q}bi->{pq}", paBI, paBI_r))
                if n != self.nmom_max:
                    pabi_r = util.einsum("pbi,bi->pbi", pabi_r, e_abi)
                    paBI_r = util.einsum("pbi,bi->pbi", paBI_r, e_aBI)

        moments_occ[0] = mpi_helper.allreduce(moments_occ[0])
        moments_occ[1] = mpi_helper.allreduce(moments_occ[1])
        moments_vir[0] = mpi_helper.allreduce(moments_vir[0])
        moments_vir[1] = mpi_helper.allreduce(moments_vir[1])

        return moments_occ, moments_vir


class BaseUGF2(BaseUGW):
    # TODO

    # --- Default GF2 options

    non_dyson = False
    compression = None

    _opts = [
        "diagonal_se",
        "non_dyson",
        "optimise_chempot",
        "fock_loop",
        "fock_opts",
        "compression",
        "compression_tol",
    ]

    def build_se_moments(self, nmom_max, integrals, **kwargs):
        """Build the moments of the self-energy.

        Parameters
        ----------
        nmom_max : int
            Maximum moment number to calculate.
        integrals : UIntegrals
            Integrals object.

        See functions in `_GF2` for `kwargs` options.

        Returns
        -------
        se_moments_hole : numpy.ndarray
            Moments of the hole self-energy for each spin channel. If
            `self.diagonal_se`, non-diagonal elements are set to zero.
        se_moments_part : numpy.ndarray
            Moments of the particle self-energy for each spin channel.
            If `self.diagonal_se`, non-diagonal elements are set to
            zero.
        """
        gf2 = _GF2(self, nmom_max, integrals, non_dyson=self.non_dyson, **kwargs)
        return gf2.kernel()


class UG0F2(BaseUGF2, UGW):
    # TODO

    _opts = BaseUGF2._opts + [opt for opt in UGW._opts if opt not in set(BaseUGW._opts)]

    @property
    def name(self):
        """Get the method name."""
        return "UG0F2"


class evUGF2(BaseUGF2, evUGW):
    # TODO

    _opts = BaseUGF2._opts + [opt for opt in evUGW._opts if opt not in set(BaseUGW._opts)]

    def __init__(self, *args, **kwargs):
        if kwargs.get("w0"):
            raise ValueError(
                f"{self.__class__.__name__} does not support option `w0`. Use of `g0` is supported "
                "for compatibility, however this is equivalent to one-shot GF2."
            )
        kwargs["w0"] = kwargs.get("g0", evUGW.g0)
        super().__init__(*args, **kwargs)

    @property
    def name(self):
        """Get the method name."""
        return f"evUG{'0' if self.g0 else ''}F2"


class UGF2(BaseUGF2, scUGW):
    # TODO

    _opts = BaseUGF2._opts + [opt for opt in scUGW._opts if opt not in set(BaseUGW._opts)]

    def __init__(self, *args, **kwargs):
        if kwargs.get("w0"):
            raise ValueError(
                f"{self.__class__.__name__} does not support option `w0`. Use of `g0` is supported "
                "for compatibility, however this is equivalent to one-shot GF2."
            )
        kwargs["w0"] = kwargs.get("g0", evUGW.g0)
        super().__init__(*args, **kwargs)

    @property
    def name(self):
        """Get the method name."""
        return f"UG{'0' if self.g0 else ''}F2"


class qsUGF2(BaseUGF2, qsUGW):
    # TODO

    _opts = BaseUGF2._opts + [opt for opt in qsUGW._opts if opt not in set(BaseUGW._opts)]

    @property
    def name(self):
        """Get the method name."""
        return f"qsUGF2"


class fsUGF2(BaseUGF2, fsUGW):
    # TODO

    _opts = BaseUGF2._opts + [opt for opt in fsUGW._opts if opt not in set(BaseUGW._opts)]

    @property
    def name(self):
        """Get the method name."""
        return f"fsUGF2"
