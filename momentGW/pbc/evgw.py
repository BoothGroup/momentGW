"""
Spin-restricted eigenvalue self-consistent GW via self-energy moment
constraints for periodic systems.
"""

import numpy as np

from momentGW import logging, util
from momentGW.evgw import evGW
from momentGW.pbc.gw import KGW


class evKGW(KGW, evGW):
    """
    Spin-restricted eigenvalue self-consistent GW via self-energy moment
    constraints for periodic systems.

    Parameters
    ----------
    mf : pyscf.pbc.scf.KSCF
        PySCF periodic mean-field class.
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
    fc : bool, optional
        If `True`, apply finite size corrections. Default value is
        `False`.
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

    _opts = util.dict_union(KGW._opts, evGW._opts)

    @property
    def name(self):
        """Get the method name."""
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

        # Get the previous moments
        if th_prev is None:
            th_prev = np.zeros_like(th)
        if tp_prev is None:
            tp_prev = np.zeros_like(tp)

        # Get the HOMO and LUMO errors
        error_homo = max(
            abs(mo[n - 1] - mo_prev[n - 1])
            for mo, mo_prev, n in zip(mo_energy, mo_energy_prev, self.nocc)
        )
        error_lumo = max(
            abs(mo[n] - mo_prev[n]) for mo, mo_prev, n in zip(mo_energy, mo_energy_prev, self.nocc)
        )

        # Get the moment errors
        error_th = max(abs(self._moment_error(t, t_prev)) for t, t_prev in zip(th, th_prev))
        error_tp = max(abs(self._moment_error(t, t_prev)) for t, t_prev in zip(tp, tp_prev))

        # Print the table
        style_homo = logging.rate(error_homo, self.conv_tol, self.conv_tol * 1e2)
        style_lumo = logging.rate(error_lumo, self.conv_tol, self.conv_tol * 1e2)
        style_th = logging.rate(error_th, self.conv_tol_moms, self.conv_tol_moms * 1e2)
        style_tp = logging.rate(error_tp, self.conv_tol_moms, self.conv_tol_moms * 1e2)
        table = logging.Table(title="Convergence")
        table.add_column("Sector", justify="right")
        table.add_column("Δ energy", justify="right")
        table.add_column("Δ moments", justify="right")
        table.add_row(
            "Hole", f"[{style_homo}]{error_homo:.3g}[/]", f"[{style_th}]{error_th:.3g}[/]"
        )
        table.add_row(
            "Particle", f"[{style_lumo}]{error_lumo:.3g}[/]", f"[{style_tp}]{error_tp:.3g}[/]"
        )
        logging.write("")
        logging.write(table)

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
