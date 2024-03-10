"""
Spin-restricted Fock matrix self-consistent GW via self-energy moment
constraints for molecular systems.
"""

import numpy as np
from pyscf.lib import logger

from momentGW import mpi_helper, util
from momentGW.base import BaseGW
from momentGW.gw import GW
from momentGW.qsgw import qsGW


def kernel(
    gw,
    nmom_max,
    mo_energy,
    mo_coeff,
    moments=None,
    integrals=None,
):
    """
    Moment-constrained Fock matrix self-consistent GW.

    Parameters
    ----------
    gw : BaseGW
        GW object.
    nmom_max : int
        Maximum moment number to calculate.
    mo_energy : numpy.ndarray
        Molecular orbital energies.
    mo_coeff : numpy.ndarray
        Molecular orbital coefficients.
    moments : tuple of numpy.ndarray, optional
        Tuple of (hole, particle) moments, if passed then they will
        be used  as the initial guess instead of calculating them.
        Default value is `None`.
    integrals : Integrals, optional
        Integrals object. If `None`, generate from scratch. Default
        value is `None`.

    Returns
    -------
    conv : bool
        Convergence flag.
    gf : dyson.Lehmann
        Green's function object
    se : dyson.Lehmann
        Self-energy object
    qp_energy : numpy.ndarray
        Quasiparticle energies.
    """

    if integrals is None:
        integrals = gw.ao2mo(transform=False)

    mo_energy = mo_energy.copy()
    mo_coeff = mo_coeff.copy()
    mo_coeff_ref = mo_coeff.copy()

    # Get the overlap
    ovlp = gw._scf.get_ovlp()

    # Get the core Hamiltonian
    h1e_ao = gw._scf.get_hcore()

    diis = util.DIIS()
    diis.space = gw.diis_space

    # Get the solver
    solver_options = {} if not gw.solver_options else gw.solver_options.copy()
    for key in gw.solver._opts:
        solver_options[key] = solver_options.get(key, getattr(gw, key, getattr(gw.solver, key)))
    subgw = gw.solver(gw._scf, **solver_options)
    subgw.verbose = 0
    integrals = subgw.ao2mo()

    conv = False
    mo_energy_prev = th_prev = tp_prev = None
    for cycle in range(1, gw.max_cycle + 1):
        logger.info(gw, "%s iteration %d", gw.name, cycle)

        # Update the Fock matrix and get the MOs
        h1e = util.einsum("...pq,...pi,...qj->...ij", h1e_ao, np.conj(mo_coeff), mo_coeff)
        dm = subgw.make_rdm1()
        fock = integrals.get_fock(dm, h1e)
        fock = gw.project_basis(fock, ovlp, mo_coeff, mo_coeff_ref)
        fock = diis.update(fock)
        mo_energy_prev = mo_energy.copy()
        mo_energy, u = np.linalg.eigh(fock)
        u = mpi_helper.bcast(u, root=0)
        mo_coeff = util.einsum("...pi,...ij->...pj", mo_coeff_ref, u)

        # Update the self-energy
        subgw.mo_energy = mo_energy
        subgw.mo_coeff = mo_coeff
        integrals = subgw.ao2mo()
        subconv, gf, se, _ = subgw.kernel(nmom_max=nmom_max, integrals=integrals)
        gf = gw.project_basis(gf, ovlp, mo_coeff, mo_coeff_ref)
        se = gw.project_basis(se, ovlp, mo_coeff, mo_coeff_ref)

        # Update the moments
        th, tp = gw.self_energy_to_moments(se, nmom_max)

        # Check for convergence
        conv = gw.check_convergence(mo_energy, mo_energy_prev, th, th_prev, tp, tp_prev)
        th_prev = th.copy()
        tp_prev = tp.copy()
        if conv:
            break

    return conv, gf, se, mo_energy


class fsGW(GW):  # noqa: D101
    __doc__ = BaseGW.__doc__.format(
        description="Spin-restricted Fock matrix self-consistent GW via self-energy moment "
        + "constraints for molecules.",
        extra_parameters="""max_cycle : int, optional
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
    solver : BaseGW, optional
        Solver to use to obtain the self-energy. Compatible with any
        `BaseGW`-like class. Default value is `momentGW.gw.GW`.
    solver_options : dict, optional
        Keyword arguments to pass to the solver. Default value is an
        empty `dict`.
    """,
    )

    # --- Default fsGW options

    fock_loop = True
    optimise_chempot = True

    # --- Extra fsGW options

    max_cycle = 50
    conv_tol = 1e-8
    conv_tol_moms = 1e-8
    conv_logical = all
    diis_space = 8
    solver = GW
    solver_options = {}

    _opts = GW._opts + [
        "max_cycle",
        "conv_tol",
        "conv_tol_moms",
        "conv_logical",
        "diis_space",
        "solver",
        "solver_options",
    ]

    @property
    def name(self):
        """Method name."""
        polarizability = self.polarizability.upper().replace("DTDA", "dTDA").replace("DRPA", "dRPA")
        return f"{polarizability}-fsGW"

    _kernel = kernel

    project_basis = staticmethod(qsGW.project_basis)
    self_energy_to_moments = staticmethod(qsGW.self_energy_to_moments)
    check_convergence = qsGW.check_convergence
