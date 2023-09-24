"""
Spin-unrestricted eigenvalue self-consistent GW via self-energy moment
constraints for periodic systems.
"""

import numpy as np
from pyscf.lib import logger

from momentGW import util
from momentGW.pbc.evgw import evKGW
from momentGW.pbc.uhf.gw import KUGW
from momentGW.uhf.evgw import evUGW


class evKUGW(KUGW, evKGW, evUGW):
    __doc__ = evKGW.__doc__.replace("Spin-restricted", "Spin-unrestricted")

    _opts = util.list_union(evKGW._opts, evKGW._opts, evUGW._opts)

    @property
    def name(self):
        """Method name."""
        polarizability = self.polarizability.upper().replace("DTDA", "dTDA").replace("DRPA", "dRPA")
        return f"{polarizability}-evKUGW"

    def check_convergence(self, mo_energy, mo_energy_prev, th, th_prev, tp, tp_prev):
        """Check for convergence, and print a summary of changes.

        Parameters
        ----------
        mo_energy : numpy.ndarray
            Molecular orbital energies at each k-point for each spin
            channel.
        mo_energy_prev : numpy.ndarray
            Molecular orbital energies from the previous iteration at
            each k-point for each spin channel.
        th : numpy.ndarray
            Moments of the occupied self-energy at each k-point for
            each spin channel.
        th_prev : numpy.ndarray
            Moments of the occupied self-energy from the previous
            iteration at each k-point for each spin channel.
        tp : numpy.ndarray
            Moments of the virtual self-energy at each k-point for each
            spin channel.
        tp_prev : numpy.ndarray
            Moments of the virtual self-energy from the previous
            iteration at each k-point for each spin channel.

        Returns
        -------
        conv : bool
            Convergence flag.
        """

        if th_prev is None:
            th_prev = np.zeros_like(th)
        if tp_prev is None:
            tp_prev = np.zeros_like(tp)

        error_homo = (
            max(
                abs(mo[n - 1] - mo_prev[n - 1])
                for mo, mo_prev, n in zip(mo_energy[0], mo_energy_prev[0], self.nocc[0])
            ),
            max(
                abs(mo[n - 1] - mo_prev[n - 1])
                for mo, mo_prev, n in zip(mo_energy[1], mo_energy_prev[1], self.nocc[1])
            ),
        )
        error_lumo = (
            max(
                abs(mo[n] - mo_prev[n])
                for mo, mo_prev, n in zip(mo_energy[0], mo_energy_prev[0], self.nocc[0])
            ),
            max(
                abs(mo[n] - mo_prev[n])
                for mo, mo_prev, n in zip(mo_energy[1], mo_energy_prev[1], self.nocc[1])
            ),
        )

        error_th = (
            max(abs(self._moment_error(t, t_prev)) for t, t_prev in zip(th[0], th_prev[0])),
            max(abs(self._moment_error(t, t_prev)) for t, t_prev in zip(th[1], th_prev[1])),
        )
        error_tp = (
            max(abs(self._moment_error(t, t_prev)) for t, t_prev in zip(tp[0], tp_prev[0])),
            max(abs(self._moment_error(t, t_prev)) for t, t_prev in zip(tp[1], tp_prev[1])),
        )

        logger.info(
            self, "Change in QPs (α): HOMO = %.6g  LUMO = %.6g", error_homo[0], error_lumo[0]
        )
        logger.info(
            self, "Change in QPs (β): HOMO = %.6g  LUMO = %.6g", error_homo[1], error_lumo[1]
        )
        logger.info(self, "Change in moments (α): occ = %.6g  vir = %.6g", error_th[0], error_tp[0])
        logger.info(self, "Change in moments (β): occ = %.6g  vir = %.6g", error_th[1], error_tp[1])

        return self.conv_logical(
            (
                max(max(error_homo), max(error_lumo)) < self.conv_tol,
                max(max(error_th), max(error_tp)) < self.conv_tol_moms,
            )
        )
