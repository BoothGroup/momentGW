"""
Base class for moment-constrained GW solvers with periodic boundary
conditions and unrestricted references.
"""

import numpy as np
from pyscf.lib import logger
from pyscf.pbc.mp.kump2 import get_frozen_mask, get_nmo, get_nocc

from momentGW.base import BaseGW
from momentGW.pbc.base import BaseKGW
from momentGW.uhf.base import BaseUGW


class BaseKUGW(BaseKGW, BaseUGW):
    __doc__ = BaseKGW.__doc__

    def kernel(
        self,
        nmom_max,
        mo_energy=None,
        mo_coeff=None,
        moments=None,
        integrals=None,
    ):
        if mo_coeff is None:
            mo_coeff = self.mo_coeff
        if mo_energy is None:
            mo_energy = self.mo_energy

        cput0 = (logger.process_clock(), logger.perf_counter())
        self.dump_flags()
        logger.info(self, "nmom_max = %d", nmom_max)

        self.converged, self.gf, self.se, self._qp_energy = self._kernel(
            nmom_max,
            mo_energy,
            mo_coeff,
            integrals=integrals,
        )

        for gf, s in zip(self.gf, ["α", "β"]):
            gf_occ = gf[0].get_occupied()
            gf_occ.remove_uncoupled(tol=1e-1)
            for n in range(min(5, gf_occ.naux)):
                en = -gf_occ.energy[-(n + 1)]
                vn = gf_occ.coupling[:, -(n + 1)]
                qpwt = np.linalg.norm(vn) ** 2
                logger.note(
                    self, "IP energy level (Γ, %s) %d E = %.16g  QP weight = %0.6g", s, n, en, qpwt
                )

        for gf, s in zip(self.gf, ["α", "β"]):
            gf_vir = gf[0].get_virtual()
            gf_vir.remove_uncoupled(tol=1e-1)
            for n in range(min(5, gf_vir.naux)):
                en = gf_vir.energy[n]
                vn = gf_vir.coupling[:, n]
                qpwt = np.linalg.norm(vn) ** 2
                logger.note(
                    self, "EA energy level (Γ, %s) %d E = %.16g  QP weight = %0.6g", s, n, en, qpwt
                )

        logger.timer(self, self.name, *cput0)

        return self.converged, self.gf, self.se, self.qp_energy

    @staticmethod
    def _gf_to_occ(gf):
        return tuple(tuple(BaseGW._gf_to_occ(g) for g in gs) for gs in gf)

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
        gf : tuple of GreensFunction
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
                    arg = np.argmax(gf[s][k].coupling[i] * gf[s][k].coupling[i].conj())
                    mo_energy[s][k][i] = gf[s][k].energy[arg]
                    check.add(arg)

                if len(check) != self.nmo[s]:
                    logger.warn(
                        self, f"Inconsistent quasiparticle weights for {spin} at k-point {k}!"
                    )

        return mo_energy

    get_nmo = get_nmo
    get_nocc = get_nocc
    get_frozen_mask = get_frozen_mask
