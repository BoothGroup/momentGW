"""
Spin-unrestricted eigenvalue self-consistent GW via self-energy moment
constraints for molecular systems.
"""

import numpy as np

from momentGW import logging, util
from momentGW import evGW
from momentGW.uhf import UGW


class evUGW(UGW, evGW):
    """
    Spin-unrestricted eigenvalue self-consistent GW via self-energy
    moment constraints for molecules.

    Parameters
    ----------
    mf : pyscf.scf.SCF
        PySCF mean-field class.
    diagonal_se : bool, optional
        If `True`, use a diagonal approximation in the self-energy.
        Default value is `False`.
    polarizability : str, optional
        Type of polarizability to use, can be one of `("drpa",
        "drpa-exact", "dtda", "thc-dtda"). Default value is `"drpa"`.
    npoints : int, optional
        Number of numerical integration points. Default value is `48`.
    optimise_chempot : bool, optional
        If `True`, optimise the chemical potential by shifting the
        position of the poles in the self-energy relative to those in
        the Green's function. Default value is `False`.
    fock_loop : bool, optional
        If `True`, self-consistently renormalise the density matrix
        according to the updated Green's function. Default value is
        `False`.
    fock_opts : dict, optional
        Dictionary of options passed to the Fock loop. For more details
        see `momentGW.fock`.
    compression : str, optional
        Blocks of the ERIs to use as a metric for compression. Can be
        one or more of `("oo", "ov", "vv", "ia")` which can be passed as
        a comma-separated string. `"oo"`, `"ov"` and `"vv"` refer to
        compression on the initial ERIs, whereas `"ia"` refers to
        compression on the ERIs entering RPA, which may change under a
        self-consistent scheme. Default value is `"ia"`.
    compression_tol : float, optional
        Tolerance for the compression. Default value is `1e-10`.
    thc_opts : dict, optional
        Dictionary of options to be used for THC calculations. Current
        implementation requires a filepath to import the THC integrals.
    g0 : bool, optional
        If `True`, do not self-consistently update the eigenvalues in
        the Green's function. Default value is `False`.
    w0 : bool, optional
        If `True`, do not self-consistently update the eigenvalues in
        the screened Coulomb interaction. Default value is `False`.
    max_cycle : int, optional
        Maximum number of iterations. Default value is `50`.
    conv_tol : float, optional
        Convergence threshold in the change in the HOMO and LUMO.
        Default value is `1e-8`.
    conv_tol_moms : float, optional
        Convergence threshold in the change in the moments. Default
        value is `1e-8`.
    conv_logical : callable, optional
        Function that takes an iterable of booleans as input indicating
        whether the individual `conv_tol` and `conv_tol_moms` have been
        satisfied, respectively, and returns a boolean indicating
        overall convergence. For example, the function `all` requires
        both metrics to be met, and `any` requires just one. Default
        value is `all`.
    diis_space : int, optional
        Size of the DIIS extrapolation space. Default value is `8`.
    damping : float, optional
        Damping parameter. Default value is `0.0`.
    weight_tol : float, optional
        Threshold in physical weight of Green's function poles, below
        which they are considered zero. Default value is `1e-11`.
    """

    _defaults = util.dict_union(UGW._defaults, evGW._defaults)

    @property
    def name(self):
        """Get the method name."""
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

        # Get the previous moments
        if th_prev is None:
            th_prev = np.zeros_like(th)
        if tp_prev is None:
            tp_prev = np.zeros_like(tp)

        # Get the HOMO and LUMO errors
        error_homo = (
            abs(mo_energy[0][self.nocc[0] - 1] - mo_energy_prev[0][self.nocc[0] - 1]),
            abs(mo_energy[1][self.nocc[1] - 1] - mo_energy_prev[1][self.nocc[1] - 1]),
        )
        error_lumo = (
            abs(mo_energy[0][self.nocc[0]] - mo_energy_prev[0][self.nocc[0]]),
            abs(mo_energy[1][self.nocc[1]] - mo_energy_prev[1][self.nocc[1]]),
        )

        # Get the moment errors
        error_th = (self._moment_error(th[0], th_prev[0]), self._moment_error(th[1], th_prev[1]))
        error_tp = (self._moment_error(tp[0], tp_prev[0]), self._moment_error(tp[1], tp_prev[1]))

        # Print the table
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
