"""
Base class for moment-constrained GW solvers with periodic boundary
conditions.
"""

import numpy as np
from pyscf.pbc.mp.kmp2 import get_frozen_mask, get_nmo, get_nocc

from momentGW import logging
from momentGW.base import Base, BaseGW
from momentGW.pbc.kpts import KPoints


class BaseKGW(BaseGW):
    """
    Base class for moment-constrained GW solvers for periodic systems.

    Parameters
    ----------
    mf : pyscf.pbc.scf.KSCF
        PySCF periodic mean-field class.
    diagonal_se : bool, optional
        If `True`, use a diagonal approximation in the self-energy.
        Default value is `False`.
    polarizability : str, optional
        Type of polarizability to use, can be one of `("drpa",
        "drpa-exact", "dtda", "thc-dtda"). Default value is `"drpa"`.
    npoints : int, optional
        Number of numerical integration points. Default value is `48`.
    optimise_chempot : bool, optional
        If `True`, optimise the chemical potential by shifting the
        position of the poles in the self-energy relative to those in
        the Green's function. Default value is `False`.
    fock_loop : bool, optional
        If `True`, self-consistently renormalise the density matrix
        according to the updated Green's function. Default value is
        `False`.
    fock_opts : dict, optional
        Dictionary of options passed to the Fock loop. For more details
        see `momentGW.pbc.fock`.
    compression : str, optional
        Blocks of the ERIs to use as a metric for compression. Can be
        one or more of `("oo", "ov", "vv", "ia")` which can be passed as
        a comma-separated string. `"oo"`, `"ov"` and `"vv"` refer to
        compression on the initial ERIs, whereas `"ia"` refers to
        compression on the ERIs entering RPA, which may change under a
        self-consistent scheme. Default value is `"ia"`.
    compression_tol : float, optional
        Tolerance for the compression. Default value is `1e-10`.
    thc_opts : dict, optional
        Dictionary of options to be used for THC calculations. Current
        implementation requires a filepath to import the THC integrals.
    fc : bool, optional
        If `True`, apply finite size corrections. Default value is
        `False`.
    """

    # --- Default KGW options

    compression = None

    # --- Extra PBC options

    fc = False
    head_wings = False

    _opts = BaseGW._opts + [
        "fc",
        "head_wings",
    ]

    get_nmo = get_nmo
    get_nocc = get_nocc
    get_frozen_mask = get_frozen_mask

    def __init__(self, mf, **kwargs):
        super().__init__(mf, **kwargs)

        # Options
        self.fc = False

        # Attributes
        self._kpts = KPoints(self.cell, getattr(mf, "kpts", np.zeros((1, 3))))

    @property
    def cell(self):
        """Get the unit cell."""
        return self._scf.cell

    @property
    def mol(self):
        """Alias for `self.cell`."""
        return self._scf.cell

    def _get_header(self):
        """
        Extend the header given by `Base._get_header` to include the
        problem size.

        Returns
        -------
        panel : rich.Table
            Panel with the solver name, options, and problem size.
        """

        # Get the options table
        options = Base._get_header(self)

        # Get the problem size table
        sizes = logging.Table(title="Sizes")
        sizes.add_column("Space", justify="right")
        sizes.add_column("Size (Γ)", justify="right")
        sizes.add_row("MOs", f"{self.nmo}")
        sizes.add_row("Occupied MOs", f"{self.nocc[0]}")
        sizes.add_row("Virtual MOs", f"{self.nmo - self.nocc[0]}")
        sizes.add_row("k-points", f"{self.kpts.kmesh} = {self.nkpts}")

        # Combine the tables
        panel = logging.Table.grid()
        panel.add_row(options)
        panel.add_row("")
        panel.add_row(sizes)

        return panel

    def _get_excitations_table(self):
        """Return the excitations as a table.

        Returns
        -------
        table : rich.Table
            Table with the excitations.
        """

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
        """
        Convert a `dyson.Lehmann` to an `mo_occ`.

        Parameters
        ----------
        gf : tuple of dyson.Lehmann
            Green's function object at each k-point.
        occupancy : int, optional
            Number of electrons in each physical orbital. Default value
            is `2`.

        Returns
        -------
        occ : tuple of numpy.ndarray
            Orbital occupation numbers at each k-point.
        """
        return tuple(BaseGW._gf_to_occ(g) for g in gf)

    @staticmethod
    def _gf_to_energy(gf):
        """
        Convert a `dyson.Lehmann` to an `mo_energy`.

        Parameters
        ----------
        gf : tuple of dyson.Lehmann
            Green's function object at each k-point.

        Returns
        -------
        energy : tuple of numpy.ndarray
            Orbital energies at each k-point.
        """
        return tuple(BaseGW._gf_to_energy(g) for g in gf)

    @staticmethod
    def _gf_to_coupling(gf, mo_coeff=None):
        """
        Convert a `dyson.Lehmann` to an `mo_coeff`.

        Parameters
        ----------
        gf : tuple of dyson.Lehmann
            Green's function object at each k-point.
        mo_coeff : numpy.ndarray, optional
            Molecular orbital coefficients at each k-point. If passed,
            rotate the Green's function couplings from the MO basis
            into the AO basis. Default value is `None`.

        Returns
        -------
        couplings : tuple of numpy.ndarray
            Couplings of the Green's function at each k-point.
        """
        if mo_coeff is None:
            mo_coeff = [None] * len(gf)
        return tuple(BaseGW._gf_to_coupling(g, mo) for g, mo in zip(gf, mo_coeff))

    def _gf_to_mo_energy(self, gf):
        """Find the poles of a GF which best overlap with the MOs.

        Parameters
        ----------
        gf : tuple of dyson.Lehmann
            Green's function object at each k-point.

        Returns
        -------
        mo_energy : numpy.ndarray
            Updated MO energies at each k-point.
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
    def kpts(self):
        """Get the k-points."""
        return self._kpts

    @property
    def nkpts(self):
        """Get the number of k-points."""
        return len(self.kpts)

    @property
    def nmo(self):
        """Get the number of molecular orbitals."""
        # PySCF returns jagged nmo with `per_kpoint=False` depending on
        # whether there is k-point dependent occupancy:
        nmo = self.get_nmo(per_kpoint=True)
        assert len(set(nmo)) == 1
        return nmo[0]

    @property
    def nocc(self):
        """Get the number of occupied molecular orbitals."""
        return self.get_nocc(per_kpoint=True)
