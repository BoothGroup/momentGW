"""
Base class for moment-constained GW solvers with unrestricted
references.
"""

import numpy as np
from pyscf.lib import logger
from pyscf.mp.ump2 import get_frozen_mask, get_nmo, get_nocc

from momentGW.base import BaseGW


class BaseUGW(BaseGW):
    def kernel(
        self,
        nmom_max,
        mo_energy=None,
        mo_coeff=None,
        moments=None,
        integrals=None,
    ):
        """Driver for the method.

        Parameters
        ----------
        nmom_max : int
            Maximum moment number to calculate.
        mo_energy : tuple of numpy.ndarray
            Molecular orbital energies for each spin channel.
        mo_coeff : tuple of numpy.ndarray
            Molecular orbital coefficients for each spin channel.
        moments : tuple of numpy.ndarray, optional
            Tuple of (hole, particle) moments for each spin channel, if
            passed then they will be used instead of calculating them.
            Default value is `None`.
        integrals : UIntegrals, optional
            Integrals object. If `None`, generate from scratch. Default
            value is `None`.
        """

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
            gf_occ = gf.get_occupied()
            gf_occ.remove_uncoupled(tol=1e-1)
            for n in range(min(5, gf_occ.naux)):
                en = -gf_occ.energy[-(n + 1)]
                vn = gf_occ.coupling[:, -(n + 1)]
                qpwt = np.linalg.norm(vn) ** 2
                logger.note(
                    self, "IP energy level (%s) %d E = %.16g  QP weight = %0.6g", s, n, en, qpwt
                )

        for gf, s in zip(self.gf, ["α", "β"]):
            gf_vir = gf.get_virtual()
            gf_vir.remove_uncoupled(tol=1e-1)
            for n in range(min(5, gf_vir.naux)):
                en = gf_vir.energy[n]
                vn = gf_vir.coupling[:, n]
                qpwt = np.linalg.norm(vn) ** 2
                logger.note(
                    self, "EA energy level (%s) %d E = %.16g  QP weight = %0.6g", s, n, en, qpwt
                )

        logger.timer(self, self.name, *cput0)

        return self.converged, self.gf, self.se, self.qp_energy

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
        gf : tuple of GreensFunction
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
                arg = np.argmax(gf[s].coupling[i] * gf[s].coupling[i].conj())
                mo_energy[s][i] = gf[s].energy[arg]
                check.add(arg)

            if len(check) != self.nmo[s]:
                logger.warn(self, f"Inconsistent quasiparticle weights for {spin}!")

        return mo_energy

    get_nmo = get_nmo
    get_nocc = get_nocc
    get_frozen_mask = get_frozen_mask
