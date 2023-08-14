"""
Spin-restricted eigenvalue self-consistent GW via self-energy moment
constraints for periodic systems.
"""

import unittest

import numpy as np
import pytest
from pyscf.agf2 import mpi_helper
from pyscf.lib import logger
from pyscf.pbc import dft, gto
from pyscf.pbc.tools import k2gamma

from momentGW.evgw import evGW, kernel
from momentGW.pbc.gw import KGW


class evKGW(KGW, evGW):
    __doc__ = evGW.__doc__.replace("molecules", "periodic systems", 1)

    @property
    def name(self):
        return "evKG%sW%s" % ("0" if self.g0 else "", "0" if self.w0 else "")

    def check_convergence(self, mo_energy, mo_energy_prev, th, th_prev, tp, tp_prev):
        """Check for convergence, and print a summary of changes."""

        if th_prev is None:
            th_prev = np.zeros_like(th)
        if tp_prev is None:
            tp_prev = np.zeros_like(tp)

        error_homo = max(
            abs(mo[n - 1] - mo_prev[n - 1])
            for mo, mo_prev, n in zip(mo_energy, mo_energy_prev, self.nocc)
        )
        error_lumo = max(
            abs(mo[n] - mo_prev[n]) for mo, mo_prev, n in zip(mo_energy, mo_energy_prev, self.nocc)
        )

        error_th = max(abs(self._moment_error(t, t_prev)) for t, t_prev in zip(th, th_prev))
        error_tp = max(abs(self._moment_error(t, t_prev)) for t, t_prev in zip(tp, tp_prev))

        logger.info(self, "Change in QPs: HOMO = %.6g  LUMO = %.6g", error_homo, error_lumo)
        logger.info(self, "Change in moments: occ = %.6g  vir = %.6g", error_th, error_tp)

        return self.conv_logical(
            (
                max(error_homo, error_lumo) < self.conv_tol,
                max(error_th, error_tp) < self.conv_tol_moms,
            )
        )

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

        self.converged, self.gf, self.se = kernel(
            self,
            nmom_max,
            mo_energy,
            mo_coeff,
            integrals=integrals,
        )

        gf_occ = self.gf[0].get_occupied()
        gf_occ.remove_uncoupled(tol=1e-1)
        for n in range(min(5, gf_occ.naux)):
            en = -gf_occ.energy[-(n + 1)]
            vn = gf_occ.coupling[:, -(n + 1)]
            qpwt = np.linalg.norm(vn) ** 2
            logger.note(self, "IP energy level (Γ) %d E = %.16g  QP weight = %0.6g", n, en, qpwt)

        gf_vir = self.gf[0].get_virtual()
        gf_vir.remove_uncoupled(tol=1e-1)
        for n in range(min(5, gf_vir.naux)):
            en = gf_vir.energy[n]
            vn = gf_vir.coupling[:, n]
            qpwt = np.linalg.norm(vn) ** 2
            logger.note(self, "EA energy level (Γ) %d E = %.16g  QP weight = %0.6g", n, en, qpwt)

        logger.timer(self, self.name, *cput0)

        return self.converged, self.gf, self.se
