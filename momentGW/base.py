"""
Base class for moment-constrained GW solvers.
"""

import numpy as np

from pyscf import lib
from pyscf.lib import logger
from pyscf.mp.mp2 import get_nmo, get_nocc, get_frozen_mask


class BaseGW(lib.StreamObject):
    """Abstract base class.

    Parameters
    ----------
    mf : pyscf.scf.SCF
        PySCF mean-field class.
    diagonal_se : bool, optional
        If `True`, use a diagonal approximation in the self-energy.
        Default value is `False`.
    polarizability : str, optional
        Type of polarizability to use, can be one of `{"drpa",
        "drpa-exact"}.  Default value is `"drpa"`.
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
    """

    # --- Default GW options

    diagonal_se = False
    polarizability = "drpa"
    optimise_chempot = False
    fock_loop = False
    fock_opts = dict(
            fock_diis_space=10,
            fock_diis_min_space=1,
            conv_tol_nelec=1e-6,
            conv_tol_rdm1=1e-8,
            max_cycle_inner=50,
            max_cycle_outer=20,
    )

    _opts = {"diagonal_se", "polarizability", "optimise_chempot"}

    def __init__(self, mf, **kwargs):
        self._scf = mf
        self.verbose = self.mol.verbose
        self.stdout = self.mol.stdout

        for key, val in kwargs.items():
            if not hasattr(self, key):
                raise AttributeError("%s has no attribute %s", self.__name__, key)
            setattr(self, key, val)

        # Do not modify:
        self.mo_coeff = mf.mo_coeff
        self.mo_occ = mf.mo_occ
        self.frozen = None
        self._nocc = None
        self._nmo = None
        self.converged = None
        self.se = None
        self.gf = None

        self._keys = set(self.__dict__.keys()).union(self._opts)

    def dump_flags(self):
        log = logger.Logger(self.stdout, self.verbose)
        log.info("")
        log.info("******** %s ********", self.__class__)
        log.info("method = %s", self.__class__.__name__)
        for key in sorted(self._opts):
            log.info("%s = %s", key, getattr(self, key))
        return self

    def build_se_static(self, *args, **kwargs):
        raise NotImplementedError

    def build_se_moments(self, *args, **kwargs):
        raise NotImplementedError

    def solve_dyson(self, *args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def _moment_error(t, t_prev):
        """Compute scaled error between moments.
        """

        error = 0
        for a, b in zip(t, t_prev):
            a = a / max(np.max(np.abs(a)), 1)
            b = b / max(np.max(np.abs(b)), 1)
            error = max(error, np.max(np.abs(a - b)))

        return error

    @staticmethod
    def _gf_to_occ(gf):
        """Convert a `GreensFunction` to an `mo_occ`.
        """

        gf_occ = gf.get_occupied()

        occ = np.zeros((gf.naux,))
        occ[:gf_occ.naux] = np.sum(np.abs(gf_occ.coupling*gf.coupling.conj()), axis=0) * 2.0

        return occ

    @property 
    def mol(self):
        return self._scf.mol

    @property
    def with_df(self):
        if getattr(self._scf, "with_df", None) is None:
            raise ValueError("GW solvers require density fitting.")
        return self._scf.with_df

    get_nmo = get_nmo
    get_nocc = get_nocc
    get_frozen_mask = get_frozen_mask

    @property
    def nmo(self):
        return self.get_nmo()

    @property
    def nocc(self):
        return self.get_nocc()
