"""
Base class for moment-constained GW solvers with unrestricted
references.
"""

import numpy as np
from pyscf.mp.ump2 import get_frozen_mask, get_nmo, get_nocc

from momentGW import logging
from momentGW.base import BaseGW


class BaseUGW(BaseGW):  # noqa: D101
    def _get_excitations_table(self):
        """Return the excitations as a table."""

        # Separate the occupied and virtual GFs
        gf_occ = (
            self.gf[0].occupied().physical(weight=1e-1),
            self.gf[0].occupied().physical(weight=1e-1),
        )
        gf_vir = (
            self.gf[1].virtual().physical(weight=1e-1),
            self.gf[1].virtual().physical(weight=1e-1),
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
                weights = gf_occ[s].couplings[:, -(n + 1)] ** 2
                weight = np.sum(weights)
                dominant = np.argsort(weights)[::-1]
                dominant = dominant[weights[dominant] > 0.1][:3]
                mo_string = ", ".join(
                    [f"{i} ({100 * weights[i] / weight:5.1f}%)" for i in dominant]
                )
                table.add_row(f"IP ({spin}) {n:>2}", f"{en:.10f}", f"{weight:.5f}", mo_string)

            # Add a break
            table.add_section()

        for s, spin in enumerate(["α", "β"]):
            # Add EAs
            for n in range(min(3, gf_vir[s].naux)):
                en = gf_vir[s].energies[n]
                weights = gf_vir[s].couplings[:, n] ** 2
                weight = np.sum(weights)
                dominant = np.argsort(weights)[::-1]
                dominant = dominant[weights[dominant] > 0.1][:3]
                mo_string = ", ".join(
                    [f"{i} ({100 * weights[i] / weight:5.1f}%)" for i in dominant]
                )
                table.add_row(f"EA ({spin}) {n:>2}", f"{en:.10f}", f"{weight:.5f}", mo_string)

            # Add a break
            if s != 1:
                table.add_section()

        return table

    @staticmethod
    def _gf_to_occ(gf):
        return tuple(BaseGW._gf_to_occ(g, occupancy=1) for g in gf)

    @staticmethod
    def _gf_to_energy(gf):
        return tuple(BaseGW._gf_to_energy(g) for g in gf)

    @staticmethod
    def _gf_to_coupling(gf, mo_coeff=None):
        if mo_coeff is None:
            mo_coeff = [None] * 2
        return tuple(BaseGW._gf_to_coupling(g, mo) for g, mo in zip(gf, mo_coeff))

    def _gf_to_mo_energy(self, gf):
        """Find the poles of a GF which best overlap with the MOs.

        Parameters
        ----------
        gf : tuple of dyson.Lehmann
            Green's function object for each spin channel.

        Returns
        -------
        mo_energy : ndarray
            Updated MO energies for each spin channel.
        """

        mo_energy = np.zeros_like(self.mo_energy)

        for s, spin in enumerate(["α", "β"]):
            check = set()
            for i in range(self.nmo[s]):
                arg = np.argmax(gf[s].couplings[i] * gf[s].couplings[i].conj())
                mo_energy[s][i] = gf[s].energies[arg]
                check.add(arg)

            if len(check) != self.nmo[s]:
                # TODO improve this warning
                logging.warn("[bad]Inconsistent quasiparticle weights![/]")

        return mo_energy

    get_nmo = get_nmo
    get_nocc = get_nocc
    get_frozen_mask = get_frozen_mask
