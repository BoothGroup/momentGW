"""
Fock matrix self-consistent loop for unrestricted references.
"""

import numpy as np
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
    self-consistent field for unrestricted references.

    Parameters
    ----------
    gw : BaseUGW
        GW object.
    gf : tuple of GreensFunction
        Green's function object for each spin channel.
    se : tuple of SelfEnergy
        Self-energy object for each spin channel.
    integrals : UIntegrals, optional
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

    h1e = lib.einsum("pq,spi,sqj->sij", gw._scf.get_hcore(), gw.mo_coeff, gw.mo_coeff)
    nmo = gw.nmo
    nocc = gw.nocc
    naux = (se[0].naux, se[1].naux)
    nqmo = (nmo[0] + naux[0], nmo[0] + naux[1])
    nelec = (nocc[0], nocc[1])

    diis = util.DIIS()
    diis.space = fock_diis_space
    diis.min_space = fock_diis_min_space
    gf_to_dm = lambda gf: np.array([g.get_occupied().moment(0) for g in gf])
    rdm1 = gf_to_dm(gf)
    fock = integrals.get_fock(rdm1, h1e)

    buf = np.zeros((max(nqmo), max(nqmo)))
    converged = False
    opts = dict(tol=conv_tol_nelec, maxiter=max_cycle_inner, occupancy=1)
    rdm1_prev = 0

    for niter1 in range(1, max_cycle_outer + 1):
        se_α, opt = minimize_chempot(se[0], fock[0], nelec[0], x0=se[0].chempot, **opts)
        se_β, opt = minimize_chempot(se[1], fock[1], nelec[1], x0=se[1].chempot, **opts)
        se = [se_α, se_β]

        for niter2 in range(1, max_cycle_inner + 1):
            w, v = se[0].eig(fock[0], chempot=0.0, out=buf)
            w = mpi_helper.bcast(w, root=0)
            v = mpi_helper.bcast(v, root=0)
            se[0].chempot, nerr_α = binsearch_chempot((w, v), nmo[0], nelec[0])

            w, v = se[1].eig(fock[1], chempot=0.0, out=buf)
            w = mpi_helper.bcast(w, root=0)
            v = mpi_helper.bcast(v, root=0)
            se[1].chempot, nerr_β = binsearch_chempot((w, v), nmo[1], nelec[1])

            w, v = se[0].eig(fock[0], out=buf)
            w = mpi_helper.bcast(w, root=0)
            v = mpi_helper.bcast(v, root=0)
            gf[0] = gf[0].__class__(w, v, chempot=se[0].chempot)

            w, v = se[1].eig(fock[1], out=buf)
            w = mpi_helper.bcast(w, root=0)
            v = mpi_helper.bcast(v, root=0)
            gf[1] = gf[1].__class__(w, v, chempot=se[1].chempot)

            rdm1 = gf_to_dm(gf)
            fock = integrals.get_fock(rdm1, h1e)
            fock = diis.update(fock, xerr=None)

            if niter2 > 1:
                derr = np.max(np.absolute(rdm1 - rdm1_prev))
                if derr < conv_tol_rdm1:
                    break

            rdm1_prev = rdm1.copy()

        logger.debug1(
            gw,
            "fock loop %d  cycles = %d  dNα = %.3g  dNβ = %.3g  |ddm| = %.3g",
            niter1,
            niter2,
            nerr_α,
            nerr_β,
            derr,
        )

        if derr < conv_tol_rdm1 and (abs(nerr_α) + abs(nerr_β)) < conv_tol_nelec:
            converged = True
            break

    logger.info(
        gw,
        "fock converged = %s  chempot (α) = %.9g  chempot (β) = %.9g  dNα = %.3g  dNβ = %.3g  |ddm| = %.3g",
        converged,
        se[0].chempot,
        se[1].chempot,
        nerr_α,
        nerr_β,
        derr,
    )

    return gf, se, converged
