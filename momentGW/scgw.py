"""
Spin-restricted self-consistent GW via self-energy moment constraitns
for molecular systems.
"""

from momentGW import logging, util
from momentGW.evgw import evGW


def kernel(
    gw,
    nmom_max,
    moments=None,
    integrals=None,
):
    """
    Moment-constrained self-consistent GW.

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
        Quasiparticle energies. Always `None` for scGW, returned for
        compatibility with other scGW methods.
    """

    if gw.polarizability.lower() == "drpa-exact":
        raise NotImplementedError("%s for polarizability=%s" % (gw.name, gw.polarizability))

    # Get the integrals
    if integrals is None:
        integrals = gw.ao2mo()

    # Initialise the orbitals and the Green's function
    mo_energy = gw.mo_energy.copy()
    gf_ref = gf = gw.init_gf(gw.mo_energy)

    # Initialise the DIIS object
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

            if cycle > 1:
                # Rotate ERIs into (MO, QMO) and (QMO occ, QMO vir)
                integrals.update_coeffs(
                    mo_coeff_g=(None if gw.g0 else gw._gf_to_coupling(gf, mo_coeff=gw.mo_coeff)),
                    mo_coeff_w=(None if gw.w0 else gw._gf_to_coupling(gf, mo_coeff=gw.mo_coeff)),
                    mo_occ_w=None if gw.w0 else gw._gf_to_occ(gf),
                )

            # Update the moments of the SE
            if moments is not None and cycle == 1:
                th, tp = moments
            else:
                th, tp = gw.build_se_moments(
                    nmom_max,
                    integrals,
                    mo_energy=dict(
                        g=gw._gf_to_energy(gf if not gw.g0 else gf_ref),
                        w=gw._gf_to_energy(gf if not gw.w0 else gf_ref),
                    ),
                    mo_occ=dict(
                        g=gw._gf_to_occ(gf if not gw.g0 else gf_ref),
                        w=gw._gf_to_occ(gf if not gw.w0 else gf_ref),
                    ),
                )

            # Extrapolate the moments
            try:
                x, xerr = gw._prepare_diis_input(th, th_prev, tp, tp_prev)
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


class scGW(evGW):
    """
    Spin-restricted self-consistent GW via self-energy moment
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
    diis_space : int, optional
        Size of the DIIS extrapolation space. Default value is `8`.
    damping : float, optional
        Damping parameter. Default value is `0.0`.
    """

    _kernel = kernel

    @property
    def name(self):
        """Get the method name."""
        return f"{self.polarizability_name}-G{'0' if self.g0 else ''}W{'0' if self.w0 else ''}"
