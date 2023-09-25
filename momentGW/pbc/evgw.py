"""
Spin-restricted eigenvalue self-consistent GW via self-energy moment
constraints for periodic systems.
"""

import numpy as np
from pyscf.lib import logger

from momentGW import util
from momentGW.evgw import evGW
from momentGW.pbc.gw import KGW


class evKGW(KGW, evGW):  # noqa: D101
    __doc__ = evGW.__doc__.replace("molecules", "periodic systems", 1)

    _opts = util.list_union(KGW._opts, evGW._opts)

    @property
    def name(self):
        """Method name."""
        polarizability = self.polarizability.upper().replace("DTDA", "dTDA").replace("DRPA", "dRPA")
        return f"{polarizability}-evKG{'0' if self.g0 else ''}W{'0' if self.w0 else ''}"

    def check_convergence(self, mo_energy, mo_energy_prev, th, th_prev, tp, tp_prev):
        """Check for convergence, and print a summary of changes.

        Parameters
        ----------
        mo_energy : numpy.ndarray
            Molecular orbital energies at each k-point.
        mo_energy_prev : numpy.ndarray
            Molecular orbital energies from the previous iteration at
            each k-point.
        th : numpy.ndarray
            Moments of the occupied self-energy at each k-point.
        th_prev : numpy.ndarray
            Moments of the occupied self-energy from the previous
            iteration at each k-point.
        tp : numpy.ndarray
            Moments of the virtual self-energy at each k-point.
        tp_prev : numpy.ndarray
            Moments of the virtual self-energy from the previous
            iteration at each k-point.

        Returns
        -------
        conv : bool
            Convergence flag.
        """

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

    def remove_unphysical_poles(self, gf):
        """
        Remove unphysical poles from the Green's function to stabilise
        iterations, according to the threshold `self.weight_tol`.

        Parameters
        ----------
        gf : tuple of dyson.Lehmann
            Green's function at each k-point.

        Returns
        -------
        gf_out : tuple of dyson.Lehmann
            Green's function at each k-point, with potentially fewer
            poles.
        """
        gf = list(gf)
        for k, g in enumerate(gf):
            gf[k] = g.physical(weight=self.weight_tol)
        return tuple(gf)
