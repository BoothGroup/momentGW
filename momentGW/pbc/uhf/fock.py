"""
Fock matrix and static self-energy parts with periodic boundary
conditions and unrestricted references.
"""

import numpy as np
from pyscf import lib
from pyscf.agf2.chempot import binsearch_chempot, minimize_chempot
from pyscf.lib import logger

from momentGW import mpi_helper, util
from momentGW.pbc.fock import minimize_chempot, search_chempot


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
    gw : BaseKUGW
        GW object.
    gf : tuple of tuple of GreensFunction
        Green's function object at each k-point for each spin channel.
    se : tuple of tuple of SelfEnergy
        Self-energy object at each k-point for each spin channel.
    integrals : KUIntegrals, optional
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

    h1e = lib.einsum("kpq,skpi,skqj->skij", gw._scf.get_hcore(), np.conj(gw.mo_coeff), gw.mo_coeff)
    nmo = gw.nmo
    nocc = gw.nocc
    naux = (
        [s.naux for s in se[0]],
        [s.naux for s in se[1]],
    )
    nqmo = (
        [nmo[0] + n for n in naux[0]],
        [nmo[1] + n for n in naux[1]],
    )
    nelec = nocc
    kpts = gw.kpts

    diis = util.DIIS()
    diis.space = fock_diis_space
    diis.min_space = fock_diis_min_space
    gf_to_dm = lambda gf: np.array([[g.get_occupied().moment(0) for g in gs] for gs in gf])
    rdm1 = gf_to_dm(gf)
    fock = integrals.get_fock(rdm1, h1e)

    buf = np.zeros((np.max(nqmo), np.max(nqmo)), dtype=complex)
    converged = False
    opts = dict(tol=conv_tol_nelec, maxiter=max_cycle_inner, occupancy=1)
    rdm1_prev = 0

    for niter1 in range(1, max_cycle_outer + 1):
        se_α, opt = minimize_chempot(se[0], fock[0], sum(nelec[0]), x0=se[0][0].chempot, **opts)
        se_β, opt = minimize_chempot(se[1], fock[1], sum(nelec[1]), x0=se[1][0].chempot, **opts)
        se = [se_α, se_β]

        for niter2 in range(1, max_cycle_inner + 1):
            w, v = zip(*[s.eig(f, chempot=0.0, out=buf) for s, f in zip(se[0], fock[0])])
            w = [mpi_helper.bcast(wk, root=0) for wk in w]
            v = [mpi_helper.bcast(vk, root=0) for vk in v]
            chempot_α, nerr_α = search_chempot(w, v, nmo[0], sum(nelec[0]), occupancy=1)

            w, v = zip(*[s.eig(f, chempot=0.0, out=buf) for s, f in zip(se[1], fock[1])])
            w = [mpi_helper.bcast(wk, root=0) for wk in w]
            v = [mpi_helper.bcast(vk, root=0) for vk in v]
            chempot_β, nerr_β = search_chempot(w, v, nmo[1], sum(nelec[1]), occupancy=1)

            for k in kpts.loop(1):
                se[0][k].chempot = chempot_α
                w, v = se[0][k].eig(fock[0][k], out=buf)
                gf[0][k] = gf[0][k].__class__(w, v[: nmo[0]], chempot=se[0][k].chempot)

                se[1][k].chempot = chempot_α
                w, v = se[1][k].eig(fock[1][k], out=buf)
                gf[1][k] = gf[1][k].__class__(w, v[: nmo[1]], chempot=se[1][k].chempot)

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
        "fock converged = %s  chempot (Γ, α) = %.9g  chempot (Γ, β) = %.9g  dNα = %.3g  dNβ = %.3g  |ddm| = %.3g",
        converged,
        se[0][0].chempot,
        se[1][0].chempot,
        nerr_α,
        nerr_β,
        derr,
    )

    return gf, se, converged
