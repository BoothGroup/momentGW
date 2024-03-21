"""
Spin-unrestricted eigenvalue self-consistent GW via self-energy moment
constraints for molecular systems.
"""

import numpy as np

from momentGW import logging, util
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

        style_homo = tuple(logging.rate(e, self.conv_tol, self.conv_tol * 1e2) for e in error_homo)
        style_lumo = tuple(logging.rate(e, self.conv_tol, self.conv_tol * 1e2) for e in error_lumo)
        style_th = tuple(
            logging.rate(e, self.conv_tol_moms, self.conv_tol_moms * 1e2) for e in error_th
        )
        style_tp = tuple(
            logging.rate(e, self.conv_tol_moms, self.conv_tol_moms * 1e2) for e in error_tp
        )
        table = logging.Table(title="Convergence")
        table.add_column("Sector", justify="right")
        table.add_column("Δ energy", justify="right")
        table.add_column("Δ moments", justify="right")
        for s, spin in enumerate(["α", "β"]):
            table.add_row(
                f"Hole ({spin})",
                f"[{style_homo[s]}]{error_homo[s]:.3g}[/]",
                f"[{style_th[s]}]{error_th[s]:.3g}[/]",
            )
        for s, spin in enumerate(["α", "β"]):
            table.add_row(
                f"Particle ({spin})",
                f"[{style_lumo[s]}]{error_lumo[s]:.3g}[/]",
                f"[{style_tp[s]}]{error_tp[s]:.3g}[/]",
            )
        logging.write("")
        logging.write(table)

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
        gf : tuple of dyson.Lehmann
            Green's function for each spin channel.

        Returns
        -------
        gf_out : tuple of dyson.Lehmann
            Green's function for each spin channel, with potentially
            fewer poles.
        """
        gf_α = gf[0].physical(weight=self.weight_tol)
        gf_β = gf[1].physical(weight=self.weight_tol)
        return (gf_α, gf_β)
