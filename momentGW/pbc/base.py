"""
Base class for moment-constrained GW solvers with periodic boundary
conditions.
"""

import numpy as np
import functools
from pyscf import lib
from pyscf.lib import logger
from pyscf.pbc.mp.kmp2 import get_frozen_mask, get_nmo, get_nocc

from momentGW.base import BaseGW


class BaseKGW(BaseGW):
    """{description}

    Parameters
    ----------
    mf : pyscf.pbc.scf.KSCF
        PySCF periodic mean-field class.
    diagonal_se : bool, optional
        If `True`, use a diagonal approximation in the self-energy.
        Default value is `False`.
    polarizability : str, optional
        Type of polarizability to use, can be one of `("drpa",
        "drpa-exact").  Default value is `"drpa"`.
    vhf_df : bool, optional
        If True, calculate the static self-energy directly from `Lpq`.
        Default value is False.
    npoints : int, optional
        Number of numerical integration points.  Default value is `48`.
    optimise_chempot : bool, optional
        If `True`, optimise the chemical potential by shifting the
        position of the poles in the self-energy relative to those in
        the Green's function.  Default value is `False`.
    fock_loop : bool, optional
        If `True`, self-consistently renormalise the density matrix
        according to the updated Green's function.  Default value is
        `False`.
    fock_opts : dict, optional
        Dictionary of options compatiable with `pyscf.dfragf2.DFRAGF2`
        objects that are used in the Fock loop.
    {extra_parameters}
    """

    @staticmethod
    def _gf_to_occ(gf):
        return tuple(BaseGW._gf_to_occ(g) for g in gf)

    @property
    def cell(self):
        return self._scf.cell

    @property
    def mol(self):
        return self.cell

    get_nmo = get_nmo
    get_nocc = get_nocc
    get_frozen_mask = get_frozen_mask

    @property
    def kpts(self):
        return self._scf.kpts

    @property
    def nkpts(self):
        return len(self.kpts)

    @property
    def nmo(self):
        nmo = self.get_nmo(per_kpoint=False)
        return nmo

    @property
    def nocc(self):
        nocc = self.get_nocc(per_kpoint=True)
        return nocc
