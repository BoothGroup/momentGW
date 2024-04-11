"""
Spin-restricted eigenvalue self-consistent GW via self-energy moment
constraints for molecular systems.
"""

import numpy as np

from momentGW import logging, util
from momentGW.gw import GW


def kernel(
    gw,
    nmom_max,
    moments=None,
    integrals=None,
):
    """
    Moment-constrained eigenvalue self-consistent GW.

    Parameters
    ----------
    gw : BaseGW
        GW object.
    nmom_max : int
        Maximum moment number to calculate.
    moments : tuple of numpy.ndarray, optional
        Tuple of (hole, particle) moments, if passed then they will
        be used  as the initial guess instead of calculating them.
        Default value is `None`.
    integrals : BaseIntegrals, optional
        Integrals object. If `None`, generate from scratch. Default
        value is `None`.

    Returns
    -------
    conv : bool
        Convergence flag.
    gf : dyson.Lehmann
        Green's function object.
    se : dyson.Lehmann
        Self-energy object.
    qp_energy : numpy.ndarray
        Quasiparticle energies. Always None for evGW, returned for
        compatibility with other evGW methods.
    """

    if gw.polarizability.lower() == "drpa-exact":
        raise NotImplementedError("%s for polarizability=%s" % (gw.name, gw.polarizability))

    # Get the integrals
    if integrals is None:
        integrals = gw.ao2mo()

    # Initialise the orbitals
    mo_energy = gw.mo_energy.copy()

    # Get the DIIS object
    diis = util.DIIS()
    diis.space = gw.diis_space

    # Get the static part of the SE
    se_static = gw.build_se_static(integrals)

    # Initialise convergence quantities
    conv = False
    th_prev = tp_prev = None

    for cycle in range(1, gw.max_cycle + 1):
        with logging.with_status(f"Iteration {cycle}"):
            with logging.with_comment(f"Start of iteration {cycle}"):
                logging.write("")

            # Update the moments of the SE
            if moments is not None and cycle == 1:
                th, tp = moments
            else:
                th, tp = gw.build_se_moments(
                    nmom_max,
                    integrals,
                    mo_energy=dict(
                        g=mo_energy if not gw.g0 else gw.mo_energy,
                        w=mo_energy if not gw.w0 else gw.mo_energy,
                    ),
                )

            # Extrapolate the moments
            try:
                x, xerr = self._prepare_diis_input(th, th_prev, tp, tp_prev)
                th, tp = diis.update(x, xerr=xerr)
            except Exception:
                logging.warn(f"DIIS step [red]failed[/] at iteration {cycle}")

            # Damp the moments
            if gw.damping != 0.0 and cycle > 1:
                th = gw.damping * th_prev + (1.0 - gw.damping) * th
                tp = gw.damping * tp_prev + (1.0 - gw.damping) * tp

            # Solve the Dyson equation
            gf, se = gw.solve_dyson(th, tp, se_static, integrals=integrals)
            gf = gw.remove_unphysical_poles(gf)

            # Update the MO energies
            mo_energy_prev = mo_energy.copy()
            mo_energy = gw._gf_to_mo_energy(gf)

            # Check for convergence
            conv = gw.check_convergence(mo_energy, mo_energy_prev, th, th_prev, tp, tp_prev)
            th_prev = th.copy()
            tp_prev = tp.copy()
            with logging.with_comment(f"End of iteration {cycle}"):
                logging.write("")
            if conv:
                break

    return conv, gf, se, None


class evGW(GW):
    """
    Spin-restricted eigenvalue self-consistent GW via self-energy moment
    constraints for molecules.

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

    # --- Extra evGW options

    g0 = False
    w0 = False
    max_cycle = 50
    conv_tol = 1e-8
    conv_tol_moms = 1e-6
    conv_logical = all
    diis_space = 8
    damping = 0.0
    weight_tol = 1e-11

    _opts = GW._opts + [
        "g0",
        "w0",
        "max_cycle",
        "conv_tol",
        "conv_tol_moms",
        "conv_logical",
        "diis_space",
        "damping",
        "weight_tol",
    ]

    _kernel = kernel

    @property
    def name(self):
        """Get the method name."""
        polarizability = self.polarizability.upper().replace("DTDA", "dTDA").replace("DRPA", "dRPA")
        return f"{polarizability}-evG{'0' if self.g0 else ''}W{'0' if self.w0 else ''}"

    def check_convergence(self, mo_energy, mo_energy_prev, th, th_prev, tp, tp_prev):
        """Check for convergence, and print a summary of changes.

        Parameters
        ----------
        mo_energy : numpy.ndarray
            Molecular orbital energies.
        mo_energy_prev : numpy.ndarray
            Molecular orbital energies from the previous iteration.
        th : numpy.ndarray
            Moments of the occupied self-energy.
        th_prev : numpy.ndarray
            Moments of the occupied self-energy from the previous
            iteration.
        tp : numpy.ndarray
            Moments of the virtual self-energy.
        tp_prev : numpy.ndarray
            Moments of the virtual self-energy from the previous iteration.

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
        error_homo = abs(mo_energy[self.nocc - 1] - mo_energy_prev[self.nocc - 1])
        error_lumo = abs(mo_energy[self.nocc] - mo_energy_prev[self.nocc])

        # Get the moment errors
        error_th = self._moment_error(th, th_prev)
        error_tp = self._moment_error(tp, tp_prev)

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
        gf : dyson.Lehmann
            Green's function object.

        Returns
        -------
        gf_out : dyson.Lehmann
            Green's function, with potentially fewer poles.
        """
        return gf.physical(weight=self.weight_tol)

    def _prepare_diis_input(self, th, th_prev, tp, tp_prev):
        """Prepare the input arrays for the DIIS extrapolation.

        Parameters
        ----------
        th : numpy.ndarray
            Moments of the occupied self-energy.
        th_prev : numpy.ndarray
            Moments of the occupied self-energy from the previous
            iteration.
        tp : numpy.ndarray
            Moments of the virtual self-energy.
        tp_prev : numpy.ndarray
            Moments of the virtual self-energy from the previous iteration.

        Returns
        -------
        x : numpy.ndarray
            Array to extrapolate using DIIS.
        xerr : numpy.ndarray
            Array of errors for the DIIS extrapolation.
        """

        # Get the array to extrapolate
        x = np.array((th, tp))

        # Get the error array
        xprev = np.array((th_prev, tp_prev)) if th_prev is not None else None
        xerr = x - xprev if xprev is not None else None

        # Scale the error array
        scale = np.max(np.abs(x), axis=-3, keepdims=True)
        scale[scale < 1e-8] = 1.0
        xerr /= scale

        return x, xerr
