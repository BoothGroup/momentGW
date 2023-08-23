"""
Fock matrix and static self-energy parts.
"""

import numpy as np
from pyscf import lib
from pyscf.agf2 import mpi_helper
from pyscf.agf2.chempot import binsearch_chempot, minimize_chempot
from pyscf.lib import logger

from momentGW import util


def fock_loop(
    gw,
    gf,
    se,
    integrals=None,
    fock_diis_space=10,
    fock_diis_min_space=1,
    conv_tol_nelec=1e-6,
    conv_tol_rdm1=1e-8,
    max_cycle_inner=100,
    max_cycle_outer=20,
):
    """
    Self-consistent loop for the density matrix via the Hartree--Fock
    self-consistent field.

    Parameters
    ----------
    gw : BaseGW
        GW object.
    gf : GreensFunction
        Green's function object.
    se : SelfEnergy
        Self-energy object.
    integrals : Integrals, optional
        Integrals object. If `None`, generate from scratch. Default
        value is `None`.
    fock_diis_space : int, optional
        DIIS space size for the Fock matrix. Default value is `10`.
    fock_diis_min_space : int, optional
        Minimum DIIS space size for the Fock matrix. Default value is
        `1`.
    conv_tol_nelec : float, optional
        Convergence tolerance for the number of electrons. Default
        value is `1e-6`.
    conv_tol_rdm1 : float, optional
        Convergence tolerance for the density matrix. Default value is
        `1e-8`.
    max_cycle_inner : int, optional
        Maximum number of inner iterations. Default value is `100`.
    max_cycle_outer : int, optional
        Maximum number of outer iterations. Default value is `20`.
    """

    if integrals is None:
        integrals = gw.ao2mo()

    h1e = np.linalg.multi_dot((gw.mo_coeff.T, gw._scf.get_hcore(), gw.mo_coeff))
    nmo = gw.nmo
    nocc = gw.nocc
    naux = se.naux
    nqmo = nmo + naux
    nelec = nocc * 2

    diis = util.DIIS()
    diis.space = fock_diis_space
    diis.min_space = fock_diis_min_space
    gf_to_dm = lambda gf: gf.get_occupied().moment(0) * 2.0
    rdm1 = gf_to_dm(gf)
    fock = integrals.get_fock(rdm1, h1e)

    buf = np.zeros((nqmo, nqmo))
    converged = False
    opts = dict(tol=conv_tol_nelec, maxiter=max_cycle_inner)
    rdm1_prev = 0

    for niter1 in range(1, max_cycle_outer + 1):
        se, opt = minimize_chempot(se, fock, nelec, x0=se.chempot, **opts)

        for niter2 in range(1, max_cycle_inner + 1):
            w, v = se.eig(fock, chempot=0.0, out=buf)
            w = mpi_helper.bcast(w, root=0)
            v = mpi_helper.bcast(v, root=0)
            se.chempot, nerr = binsearch_chempot((w, v), nmo, nelec)

            w, v = se.eig(fock, out=buf)
            w = mpi_helper.bcast(w, root=0)
            v = mpi_helper.bcast(v, root=0)
            gf = gf.__class__(w, v[:nmo], chempot=se.chempot)

            rdm1 = gf_to_dm(gf)
            fock = integrals.get_fock(rdm1, h1e)
            fock = diis.update(fock, xerr=None)

            if niter2 > 1:
                derr = np.max(np.absolute(rdm1 - rdm1_prev))
                if derr < conv_tol_rdm1:
                    break

            rdm1_prev = rdm1.copy()

        logger.debug1(
            gw, "fock loop %d  cycles = %d  dN = %.3g  |ddm| = %.3g", niter1, niter2, nerr, derr
        )

        if derr < conv_tol_rdm1 and abs(nerr) < conv_tol_nelec:
            converged = True
            break

    logger.info(
        gw,
        "fock converged = %s  chempot = %.9g  dN = %.3g  |ddm| = %.3g",
        converged,
        se.chempot,
        nerr,
        derr,
    )

    return gf, se, converged
