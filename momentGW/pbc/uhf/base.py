"""
Base class for moment-constrained GW solvers with periodic boundary
conditions and unrestricted references.
"""

import numpy as np
from pyscf.pbc.mp.kump2 import get_frozen_mask, get_nmo, get_nocc

from momentGW import logging
from momentGW.base import Base, BaseGW
from momentGW.pbc.base import BaseKGW
from momentGW.uhf.base import BaseUGW


class BaseKUGW(BaseKGW, BaseUGW):  # noqa: D101
    __doc__ = BaseKGW.__doc__

    get_nmo = get_nmo
    get_nocc = get_nocc
    get_frozen_mask = get_frozen_mask

    def _get_header(self):
        """
        Get the header for the solver, with the name, options, and
        problem size.
        """

        # Get the options table
        options = Base._get_header(self)

        # Get the problem size table
        sizes = logging.Table(title="Sizes")
        sizes.add_column("Space", justify="right")
        sizes.add_column("Size (Γ, α)", justify="right")
        sizes.add_column("Size (Γ, β)", justify="right")
        sizes.add_row("MOs", f"{self.nmo[0]}", f"{self.nmo[1]}")
        sizes.add_row("Occupied MOs", f"{self.nocc[0][0]}", f"{self.nocc[1][0]}")
        sizes.add_row(
            "Virtual MOs", f"{self.nmo[0] - self.nocc[0][0]}", f"{self.nmo[1] - self.nocc[1][0]}"
        )
        sizes.add_row("k-points", f"{self.kpts.kmesh} = {self.nkpts}")

        # Combine the tables
        panel = logging.Table.grid()
        panel.add_row(options)
        panel.add_row("")
        panel.add_row(sizes)

        return panel

    def _get_excitations_table(self):
        """Return the excitations as a table."""

        # Separate the occupied and virtual GFs
        gf_occ = (
            self.gf[0][0].occupied().physical(weight=1e-1),
            self.gf[1][0].occupied().physical(weight=1e-1),
        )
        gf_vir = (
            self.gf[0][0].virtual().physical(weight=1e-1),
            self.gf[1][0].virtual().physical(weight=1e-1),
        )

        # Build table
        table = logging.Table(title="Green's function poles")
        table.add_column("Excitation", justify="right")
        table.add_column("Energy", justify="right", style="output")
        table.add_column("QP weight", justify="right")
        table.add_column("Dominant MOs", justify="right")

        # Add IPs
        for s, spin in enumerate(["α", "β"]):
            for n in range(min(3, gf_occ[s].naux)):
                en = -gf_occ[s].energies[-(n + 1)]
                weights = np.real(
                    gf_occ[s].couplings[:, -(n + 1)] * gf_occ[s].couplings[:, -(n + 1)].conj()
                )
                weight = np.sum(weights)
                dominant = np.argsort(weights)[::-1]
                dominant = dominant[weights[dominant] > 0.1][:3]
                mo_string = ", ".join(
                    [f"{i} ({100 * weights[i] / weight:5.1f}%)" for i in dominant]
                )
                table.add_row(f"IP (Γ, {spin}) {n:>2}", f"{en:.10f}", f"{weight:.5f}", mo_string)

            # Add a break
            table.add_section()

        for s, spin in enumerate(["α", "β"]):
            # Add EAs
            for n in range(min(3, gf_vir[s].naux)):
                en = gf_vir[s].energies[n]
                weights = np.real(gf_vir[s].couplings[:, n] * gf_vir[s].couplings[:, n].conj())
                weight = np.sum(weights)
                dominant = np.argsort(weights)[::-1]
                dominant = dominant[weights[dominant] > 0.1][:3]
                mo_string = ", ".join(
                    [f"{i} ({100 * weights[i] / weight:5.1f}%)" for i in dominant]
                )
                table.add_row(f"EA (Γ, {spin}) {n:>2}", f"{en:.10f}", f"{weight:.5f}", mo_string)

            # Add a break
            if s != 1:
                table.add_section()

        return table

    @staticmethod
    def _gf_to_occ(gf):
        return tuple(tuple(BaseGW._gf_to_occ(g, occupancy=1) for g in gs) for gs in gf)

    @staticmethod
    def _gf_to_energy(gf):
        return tuple(tuple(BaseGW._gf_to_energy(g) for g in gs) for gs in gf)

    @staticmethod
    def _gf_to_coupling(gf, mo_coeff=None):
        if mo_coeff is None:
            mo_coeff = [[None] * len(gf[0])] * 2
        return tuple(
            tuple(BaseGW._gf_to_coupling(g, mo) for g, mo in zip(gs, mos))
            for gs, mos in zip(gf, mo_coeff)
        )

    def _gf_to_mo_energy(self, gf):
        """Find the poles of a GF which best overlap with the MOs.

        Parameters
        ----------
        gf : tuple of dyson.Lehmann
            Green's function object for each k-point for each spin
            channel.

        Returns
        -------
        mo_energy : ndarray
            Updated MO energies for each k-point for each spin channel.
        """

        mo_energy = np.zeros_like(self.mo_energy)

        for s, spin in enumerate(["α", "β"]):
            for k in self.kpts.loop(1):
                check = set()
                for i in range(self.nmo[s]):
                    arg = np.argmax(gf[s][k].couplings[i] * gf[s][k].couplings[i].conj())
                    mo_energy[s][k][i] = gf[s][k].energies[arg]
                    check.add(arg)

                if len(check) != self.nmo[s]:
                    # TODO improve this warning
                    logging.warn(
                        f"[bad]Inconsistent quasiparticle weights for {spin} at k-point {k}![/]"
                    )

        return mo_energy

    @property
    def nmo(self):
        """Return the number of molecular orbitals."""
        # PySCF returns jagged nmo with `per_kpoint=False` depending on
        # whether there is k-point dependent occupancy:
        nmo = self.get_nmo(per_kpoint=True)
        assert len(set(nmo[0])) == 1
        assert len(set(nmo[1])) == 1
        return nmo[0][0], nmo[1][0]
