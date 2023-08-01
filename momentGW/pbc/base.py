"""
Base class for moment-constrained GW solvers with periodic boundary
conditions.
"""

import functools

import numpy as np
from pyscf import lib
from pyscf.lib import logger
from pyscf.pbc.mp.kmp2 import get_frozen_mask, get_nmo, get_nocc

from momentGW.base import BaseGW
from momentGW.pbc.kpts import KPoints


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
    compression : str, optional
        Blocks of the ERIs to use as a metric for compression. Can be
        one or more of `("oo", "ov", "vv", "ia")` which can be passed as
        a comma-separated string. `"oo"`, `"ov"` and `"vv"` refer to
        compression on the initial ERIs, whereas `"ia"` refers to
        compression on the ERIs entering RPA, which may change under a
        self-consistent scheme.  Default value is `"ia"`.
    compression_tol : float, optional
        Tolerance for the compression.  Default value is `1e-10`.
    {extra_parameters}
    """

    def __init__(self, mf, **kwargs):
        self._scf = mf
        self.verbose = self.mol.verbose
        self.stdout = self.mol.stdout
        self.max_memory = 1e10

        for key, val in kwargs.items():
            if not hasattr(self, key):
                raise AttributeError("%s has no attribute %s", self.name, key)
            setattr(self, key, val)

        # Do not modify:
        self.mo_energy = mf.mo_energy
        self.mo_coeff = mf.mo_coeff
        self.mo_occ = mf.mo_occ
        self.frozen = None
        self._nocc = None
        self._nmo = None
        self._kpts = KPoints(self.cell, mf.kpts)
        self.converged = None
        self.se = None
        self.gf = None

        self._keys = set(self.__dict__.keys()).union(self._opts)

    @staticmethod
    def _gf_to_occ(gf):
        return tuple(BaseGW._gf_to_occ(g) for g in gf)

    @property
    def cell(self):
        return self._scf.cell

    mol = cell

    get_nmo = get_nmo
    get_nocc = get_nocc
    get_frozen_mask = get_frozen_mask

    @property
    def kpts(self):
        return self._kpts

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
