"""
Spin-unrestricted eigenvalue self-consistent GW via self-energy moment
constraints for molecular systems.
"""

import numpy as np
from pyscf.lib import logger

from momentGW import util
from momentGW.evgw import evGW
from momentGW.uhf import UGW


class evUGW(UGW, evGW):  # noqa: D101
    __doc__ = evGW.__doc__.replace("Spin-restricted", "Spin-unrestricted", 1)

    _opts = util.list_union(UGW._opts, evGW._opts)

    @property
    def name(self):
        """Method name."""
        polarizability = self.polarizability.upper().replace("DTDA", "dTDA").replace("DRPA", "dRPA")
        return f"{polarizability}-evUG{'0' if self.g0 else ''}W{'0' if self.w0 else ''}"

    def check_convergence(self, mo_energy, mo_energy_prev, th, th_prev, tp, tp_prev):
        """Check for convergence, and print a summary of changes.

        Parameters
        ----------
        mo_energy : numpy.ndarray
            Molecular orbital energies for each spin channel.
        mo_energy_prev : numpy.ndarray
            Molecular orbital energies from the previous iteration for
            each spin channel.
        th : numpy.ndarray
            Moments of the occupied self-energy for each spin channel.
        th_prev : numpy.ndarray
            Moments of the occupied self-energy from the previous
            iteration for each spin channel.
        tp : numpy.ndarray
            Moments of the virtual self-energy for each spin channel.
        tp_prev : numpy.ndarray
            Moments of the virtual self-energy from the previous
            iteration for each spin channel.

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
            abs(mo_energy[0][self.nocc[0] - 1] - mo_energy_prev[0][self.nocc[0] - 1]),
            abs(mo_energy[1][self.nocc[1] - 1] - mo_energy_prev[1][self.nocc[1] - 1]),
        )
        error_lumo = (
            abs(mo_energy[0][self.nocc[0]] - mo_energy_prev[0][self.nocc[0]]),
            abs(mo_energy[1][self.nocc[1]] - mo_energy_prev[1][self.nocc[1]]),
        )

        error_th = (self._moment_error(th[0], th_prev[0]), self._moment_error(th[1], th_prev[1]))
        error_tp = (self._moment_error(tp[0], tp_prev[0]), self._moment_error(tp[1], tp_prev[1]))

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

    def remove_unphysical_poles(self, gf):
        """
        Remove unphysical poles from the Green's function to stabilise
        iterations, according to the threshold `self.weight_tol`.

        Parameters
        ----------
        gf : tuple of Lehmann
            Green's function for each spin channel.

        Returns
        -------
        gf_out : tuple of Lehmann
            Green's function for each spin channel, with potentially
            fewer poles.
        """
        gf_α = gf[0].physical(weight=self.weight_tol)
        gf_β = gf[1].physical(weight=self.weight_tol)
        return (gf_α, gf_β)
