"""
Base class for moment-constrained GW solvers.
"""

import warnings

import numpy as np
from pyscf import lib
from pyscf.agf2 import mpi_helper
from pyscf.lib import logger
from pyscf.mp.mp2 import get_frozen_mask, get_nmo, get_nocc


class BaseGW(lib.StreamObject):
    """{description}

    Parameters
    ----------
    mf : pyscf.scf.SCF
        PySCF mean-field class.
    diagonal_se : bool, optional
        If `True`, use a diagonal approximation in the self-energy.
        Default value is `False`.
    polarizability : str, optional
        Type of polarizability to use, can be one of `("drpa",
        "drpa-exact", "dtda").  Default value is `"drpa"`.
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

    # --- Default GW options

    diagonal_se = False
    polarizability = "drpa"
    npoints = 48
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
    compression = "ia"
    compression_tol = 1e-10

    _opts = [
        "diagonal_se",
        "polarizability",
        "npoints",
        "optimise_chempot",
        "fock_loop",
        "fock_opts",
        "compression",
        "compression_tol",
    ]

    def __init__(self, mf, **kwargs):
        self._scf = mf
        self.verbose = self.mol.verbose
        self.stdout = self.mol.stdout
        self.max_memory = 1e10

        if kwargs.pop("vhf_df", None) is not None:
            warnings.warn("Keyword argument vhf_df is deprecated.", DeprecationWarning)

        for key, val in kwargs.items():
            if not hasattr(self, key):
                raise AttributeError("%s has no attribute %s", self.name, key)
            setattr(self, key, val)

        # Do not modify:
        self.mo_energy = mpi_helper.bcast(mf.mo_energy, root=0)
        self.mo_coeff = mpi_helper.bcast(mf.mo_coeff, root=0)
        self.mo_occ = mf.mo_occ
        self.frozen = None
        self._nocc = None
        self._nmo = None
        self.converged = None
        self.se = None
        self.gf = None
        self._qp_energy = None

        self._keys = set(self.__dict__.keys()).union(self._opts)

    def dump_flags(self):
        log = logger.Logger(self.stdout, self.verbose)
        log.info("")
        log.info("******** %s ********", self.__class__)
        log.info("method = %s", self.name)
        for key in self._opts:
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
        """Compute scaled error between moments."""

        if t_prev is None:
            t_prev = np.zeros_like(t)

        error = 0
        for a, b in zip(t, t_prev):
            a = a / max(np.max(np.abs(a)), 1)
            b = b / max(np.max(np.abs(b)), 1)
            error = max(error, np.max(np.abs(a - b)))

        return error

    @staticmethod
    def _gf_to_occ(gf):
        """Convert a `GreensFunction` to an `mo_occ`."""

        gf_occ = gf.get_occupied()

        occ = np.zeros((gf.naux,))
        occ[: gf_occ.naux] = np.sum(np.abs(gf_occ.coupling * gf_occ.coupling.conj()), axis=0) * 2.0

        return occ

    def _gf_to_mo_energy(self, gf):
        """Find the poles of a GF which best overlap with the MOs.

        Parameters
        ----------
        gf : GreensFunction
            Green's function object.

        Returns
        -------
        mo_energy : ndarray
            Updated MO energies.
        """

        check = set()
        mo_energy = np.zeros_like(self.mo_energy)

        for i in range(self.nmo):
            arg = np.argmax(gf.coupling[i] ** 2)
            mo_energy[i] = gf.energy[arg]
            check.add(arg)

        if len(check) != self.nmo:
            logger.warn(self, "Inconsistent quasiparticle weights!")

        return mo_energy

    @property
    def qp_energy(self):
        """
        Return the quasiparticle energies. For most GW methods, this
        simply consists of the poles of the `self.gf` that best
        overlap with the MOs, in order. In some methods such as qsGW,
        these two quantities are not the same.
        """

        if self._qp_energy is not None:
            return self._qp_energy

        qp_energy = self._gf_to_mo_energy(self.gf)

        return qp_energy

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
