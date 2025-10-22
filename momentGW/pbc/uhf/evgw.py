"""Spin-unrestricted eigenvalue self-consistent GW via self-energy moment constraints for periodic
systems.
"""

import numpy as np

from momentGW import logging, util
from momentGW.pbc.evgw import evKGW
from momentGW.pbc.uhf.gw import KUGW
from momentGW.uhf.evgw import evUGW


class evKUGW(KUGW, evKGW, evUGW):
    """Spin-unrestricted eigenvalue self-consistent GW via self-energy moment constraints for
    periodic systems.

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

    _defaults = util.dict_union(evKGW._defaults, evKGW._defaults, evUGW._defaults)

    @property
    def name(self):
        """Get the method name."""
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

        # Get the previous moments
        if th_prev is None:
            th_prev = np.zeros_like(th)
        if tp_prev is None:
            tp_prev = np.zeros_like(tp)

        def try_index(x, n):
            if n < 0 or n >= len(x):
                return 0.0
            else:
                return x[n]

        # Get the HOMO and LUMO errors
        error_homo = (
            max(
                abs(try_index(mo, n - 1) - try_index(mo_prev, n - 1))
                for mo, mo_prev, n in zip(mo_energy[0], mo_energy_prev[0], self.nocc[0])
            ),
            max(
                abs(try_index(mo, n - 1) - try_index(mo_prev, n - 1))
                for mo, mo_prev, n in zip(mo_energy[1], mo_energy_prev[1], self.nocc[1])
            ),
        )
        error_lumo = (
            max(
                abs(try_index(mo, n) - try_index(mo_prev, n))
                for mo, mo_prev, n in zip(mo_energy[0], mo_energy_prev[0], self.nocc[0])
            ),
            max(
                abs(try_index(mo, n) - try_index(mo_prev, n))
                for mo, mo_prev, n in zip(mo_energy[1], mo_energy_prev[1], self.nocc[1])
            ),
        )

        # Get the moment errors
        error_th = (
            max(abs(self._moment_error(t, t_prev)) for t, t_prev in zip(th[0], th_prev[0])),
            max(abs(self._moment_error(t, t_prev)) for t, t_prev in zip(th[1], th_prev[1])),
        )
        error_tp = (
            max(abs(self._moment_error(t, t_prev)) for t, t_prev in zip(tp[0], tp_prev[0])),
            max(abs(self._moment_error(t, t_prev)) for t, t_prev in zip(tp[1], tp_prev[1])),
        )

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
        """Remove unphysical poles from the Green's function to stabilise iterations, according to
        the threshold `self.weight_tol`.

        Parameters
        ----------
        gf : tuple of tuple of dyson.Lehmann
            Green's function at each k-point for each spin channel.

        Returns
        -------
        gf_out : tuple of tuple of dyson.Lehmann
            Green's function at each k-point for each spin channel, with
            potentially fewer poles.
        """
        gf = [[g for g in gs] for gs in gf]
        for s in range(2):
            for k, g in enumerate(gf[s]):
                gf[s][k] = g.physical(weight=self.weight_tol)
        return (tuple(gf[0]), tuple(gf[1]))
