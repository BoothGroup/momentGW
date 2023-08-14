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

from momentGW.evgw import evGW
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
