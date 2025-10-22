"""Spin-restricted Fock matrix self-consistent GW via self-energy moment
constraints for molecular systems.
"""

import copy
from collections import OrderedDict

import numpy as np

from momentGW import logging, mpi_helper, util
from momentGW.gw import GW
from momentGW.qsgw import qsGW


def kernel(
    gw,
    nmom_max,
    moments=None,
    integrals=None,
):
    """Moment-constrained Fock matrix self-consistent GW.

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
        Quasiparticle energies.
    """

    # Get the integrals
    if integrals is None:
        integrals = gw.ao2mo()

    # Initialise the orbitals
    mo_energy = gw.mo_energy.copy()
    mo_coeff = gw.mo_coeff.copy()

    with util.SilentSCF(gw._scf):
        # Get the overlap
        ovlp = gw._scf.get_ovlp()

        # Get the core Hamiltonian
        h1e_ao = gw._scf.get_hcore()
        h1e = util.einsum("...pq,...pi,...qj->...ij", h1e_ao, np.conj(gw.mo_coeff), gw.mo_coeff)

    # Initialise the DIIS object
    diis = util.DIIS()
    diis.space = gw.diis_space

    # Get the solver
    solver_options = {} if not gw.solver_options else gw.solver_options.copy()
    for key in gw.solver._defaults:
        if key not in solver_options:
            solver_options[key] = copy.deepcopy(gw._opts.get(key, gw.solver._defaults[key]))
    with logging.with_silent():
        subgw = gw.solver(gw._scf, **solver_options)
        subgw.frozen = gw.frozen
        gf = subgw.init_gf()

    # Initialise convergence quantities
    conv = False
    mo_energy_prev = th_prev = tp_prev = None

    for cycle in range(1, gw.max_cycle + 1):
        with logging.with_comment(f"Start of iteration {cycle}"):
            logging.write("")

        with logging.with_status(f"Iteration {cycle}"):
            # Update the Fock matrix
            dm = subgw.make_rdm1(gf=gf)
            fock = integrals.get_fock(dm, h1e)
            fock = diis.update(fock)

            # Update the MOs
            mo_energy_prev = mo_energy.copy()
            mo_energy, u = np.linalg.eigh(fock)
            u = mpi_helper.bcast(u, root=0)
            mo_coeff = util.einsum("...pi,...ij->...pj", gw.mo_coeff, u)

            # Update the self-energy
            mo_energy_full = gw.mo_energy_with_frozen.copy()
            mo_energy_full[..., gw.active] = mo_energy
            subgw.mo_energy = mo_energy_full
            mo_coeff_full = gw.mo_coeff_with_frozen.copy()
            mo_coeff_full[..., gw.active] = mo_coeff
            subgw.mo_coeff = mo_coeff_full
            subconv, gf, se, _ = subgw._kernel(nmom_max)
            gf = gw.project_basis(gf, ovlp, mo_coeff, gw.mo_coeff)
            se = gw.project_basis(se, ovlp, mo_coeff, gw.mo_coeff)

            # Update the moments
            th, tp = gw.self_energy_to_moments(se, nmom_max)

            # Damp the moments
            if gw.damping != 0.0 and cycle > 1:
                th = gw.damping * th_prev + (1.0 - gw.damping) * th
                tp = gw.damping * tp_prev + (1.0 - gw.damping) * tp

            # Check for convergence
            conv = gw.check_convergence(mo_energy, mo_energy_prev, th, th_prev, tp, tp_prev)
            th_prev = th.copy()
            tp_prev = tp.copy()
            with logging.with_comment(f"End of iteration {cycle}"):
                logging.write("")
            if conv:
                break

    return conv, gf, se, mo_energy


class fsGW(GW):
    """Spin-restricted Fock matrix self-consistent GW via self-energy
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
        whether the individual `conv_tol`, `conv_tol_moms` have been
        satisfied, respectively, and returns a boolean indicating
        overall convergence. For example, the function `all` requires
        both metrics to be met, and `any` requires just one. Default
        value is `all`.
    diis_space : int, optional
        Size of the DIIS extrapolation space. Default value is `8`.
    damping : float, optional
        Damping parameter. Default value is `0.0`.
    solver : BaseGW, optional
        Solver to use to obtain the self-energy. Compatible with any
        `BaseGW`-like class. Default value is `momentGW.gw.GW`.
    solver_options : dict, optional
        Keyword arguments to pass to the solver. Default value is an
        empty `dict`.
    """

    _defaults = OrderedDict(
        **GW._defaults,
        max_cycle=50,
        conv_tol=1e-8,
        conv_tol_moms=1e-8,
        conv_logical=all,
        diis_space=8,
        damping=0.0,
        solver=GW,
        solver_options={},
    )
    _defaults["fock_loop"] = True
    _defaults["optimise_chempot"] = True

    _kernel = kernel

    project_basis = staticmethod(qsGW.project_basis)
    self_energy_to_moments = staticmethod(qsGW.self_energy_to_moments)
    check_convergence = qsGW.check_convergence

    @property
    def name(self):
        """Get the method name."""
        polarizability = self.polarizability.upper().replace("DTDA", "dTDA").replace("DRPA", "dRPA")
        return f"{polarizability}-fsGW"
