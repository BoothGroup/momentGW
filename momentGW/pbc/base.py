"""
Base class for moment-constrained GW solvers with periodic boundary
conditions.
"""

import numpy as np
from pyscf.pbc.mp.kmp2 import get_frozen_mask, get_nmo, get_nocc

from momentGW import logging
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

    # --- Default KGW options

    compression = None

    # --- Extra PBC options

    fc = False

    _opts = BaseGW._opts + [
        "fc",
    ]

    def __init__(self, mf, **kwargs):
        super().__init__(mf, **kwargs)

        # Options
        self.fc = False

        # Attributes
        self._kpts = KPoints(self.cell, getattr(mf, "kpts", np.zeros((1, 3))))

    def _get_excitations_table(self):
        """Return the excitations as a table."""

        # Separate the occupied and virtual GFs
        gf_occ = self.gf[0].occupied().physical(weight=1e-1)
        gf_vir = self.gf[0].virtual().physical(weight=1e-1)

        # Build table
        table = logging.Table(title="Green's function poles")
        table.add_column("Excitation", justify="right")
        table.add_column("Energy", justify="right", style="output")
        table.add_column("QP weight", justify="right")
        table.add_column("Dominant MOs", justify="right")

        # Add IPs
        for n in range(min(3, gf_occ.naux)):
            en = -gf_occ.energies[-(n + 1)]
            weights = np.real(gf_occ.couplings[:, -(n + 1)] * gf_occ.couplings[:, -(n + 1)].conj())
            weight = np.sum(weights)
            dominant = np.argsort(weights)[::-1]
            dominant = dominant[weights[dominant] > 0.1][:3]
            mo_string = ", ".join([f"{i} ({100 * weights[i] / weight:5.1f}%)" for i in dominant])
            table.add_row(f"IP (Γ) {n:>2}", f"{en:.10f}", f"{weight:.5f}", mo_string)

        # Add a break
        table.add_section()

        # Add EAs
        for n in range(min(3, gf_vir.naux)):
            en = gf_vir.energies[n]
            weights = np.real(gf_vir.couplings[:, n] * gf_vir.couplings[:, n].conj())
            weight = np.sum(weights)
            dominant = np.argsort(weights)[::-1]
            dominant = dominant[weights[dominant] > 0.1][:3]
            mo_string = ", ".join([f"{i} ({100 * weights[i] / weight:5.1f}%)" for i in dominant])
            table.add_row(f"EA (Γ) {n:>2}", f"{en:.10f}", f"{weight:.5f}", mo_string)

        return table

    @staticmethod
    def _gf_to_occ(gf):
        return tuple(BaseGW._gf_to_occ(g) for g in gf)

    @staticmethod
    def _gf_to_energy(gf):
        return tuple(BaseGW._gf_to_energy(g) for g in gf)

    @staticmethod
    def _gf_to_coupling(gf, mo_coeff=None):
        if mo_coeff is None:
            mo_coeff = [None] * len(gf)
        return tuple(BaseGW._gf_to_coupling(g, mo) for g, mo in zip(gf, mo_coeff))

    def _gf_to_mo_energy(self, gf):
        """Find the poles of a GF which best overlap with the MOs.

        Parameters
        ----------
        gf : tuple of dyson.Lehmann
            Green's function object for each k-point.

        Returns
        -------
        mo_energy : ndarray
            Updated MO energies for each k-point.
        """

        mo_energy = np.zeros_like(self.mo_energy)

        for k in self.kpts.loop(1):
            check = set()
            for i in range(self.nmo):
                arg = np.argmax(gf[k].couplings[i] * gf[k].couplings[i].conj())
                mo_energy[k][i] = gf[k].energies[arg]
                check.add(arg)

            if len(check) != self.nmo:
                # TODO improve this warning
                logging.warn(f"[bad]Inconsistent quasiparticle weights at k-point {k}![/]")

        return mo_energy

    @property
    def cell(self):
        """Return the unit cell."""
        return self._scf.cell

    mol = cell

    get_nmo = get_nmo
    get_nocc = get_nocc
    get_frozen_mask = get_frozen_mask

    @property
    def kpts(self):
        """Return the k-points."""
        return self._kpts

    @property
    def nkpts(self):
        """Return the number of k-points."""
        return len(self.kpts)

    @property
    def nmo(self):
        """Return the number of molecular orbitals."""
        # PySCF returns jagged nmo with `per_kpoint=False` depending on
        # whether there is k-point dependent occupancy:
        nmo = self.get_nmo(per_kpoint=True)
        assert len(set(nmo)) == 1
        return nmo[0]

    @property
    def nocc(self):
        """
        Return the number of occupied molecular orbitals at each k-point.
        """
        nocc = self.get_nocc(per_kpoint=True)
        return nocc
