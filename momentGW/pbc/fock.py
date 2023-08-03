"""
Fock matrix and static self-energy parts with periodic boundary
conditions.
"""

import numpy as np
from pyscf import lib
from pyscf.lib import logger
from pyscf.agf2.chempot import binsearch_chempot, minimize_chempot

from momentGW import util


def get_j(Lpq, dm, kpts):
    """
    Build the J matrix.
    """

    nkpts, nmo, _ = dm.shape
    vj = np.zeros_like(dm)

    for (ki, kpti), (kk, kptk) in kpts.loop(2):
        kj = ki
        kl = kpts.conserve(ki, kj, kk)
        buf = lib.einsum("Lpq,pq->L", Lpq[kk, kl], dm[kl])
        vj[ki] += lib.einsum("Lpq,L->pq", Lpq[ki, kj], buf)

    vj /= len(kpts)

    return vj


def get_k(Lpq, dm, kpts):
    """
    Build the K matrix.
    """

    nkpts, nmo, _ = dm.shape
    vk = np.zeros_like(dm)

    for (ki, kpti), (kk, kptk) in kpts.loop(2):
        kj = ki
        kl = kpts.conserve(ki, kj, kk)
        buf = np.dot(Lpq[ki, kl].reshape(-1, nmo), dm[kl].conj())
        buf = buf.reshape(-1, nmo, nmo).swapaxes(1, 2).reshape(-1, nmo)
        vk[ki] += np.dot(buf.T, Lpq[kk, kj].reshape(-1, nmo)).T.conj()

    vk /= len(kpts)

    return vk


def get_jk(Lpq, dm, kpts):
    return get_j(Lpq, dm, kpts), get_k(Lpq, dm, kpts)


def get_fock(Lpq, dm, h1e, kpts):
    vj, vk = get_jk(Lpq, dm, kpts)
    return h1e + vj - vk * 0.5


def fock_loop(
    gw,
    Lpq,
    gf,
    se,
    fock_diis_space=10,
    fock_diis_min_space=1,
    conv_tol_nelec=1e-6,
    conv_tol_rdm1=1e-8,
    max_cycle_inner=100,
    max_cycle_outer=20,
):
    """Self-consistent loop for the density matrix via the HF self-
    consistent field.
    """

    h1e = lib.einsum("kpq,kpi,kqj->kij", gw._scf.get_hcore(), gw.mo_coeff, np.conj(gw.mo_coeff))
    nmo = gw.nmo
    nocc = gw.nocc
    naux = [s.naux for s in se]
    nqmo = [nmo + n for n in naux]
    nelec = [n * 2 for n in nocc]
    kpts = gw.kpts

    diis = util.DIIS()
    diis.space = fock_diis_space
    diis.min_space = fock_diis_min_space
    gf_to_dm = lambda gf: np.array([g.get_occupied().moment(0) for g in gf]) * 2.0
    rdm1 = gf_to_dm(gf)
    fock = get_fock(Lpq, rdm1, h1e, kpts)

    buf = np.zeros((max(nqmo), max(nqmo)), dtype=complex)
    converged = False
    opts = dict(tol=conv_tol_nelec, maxiter=max_cycle_inner)
    rdm1_prev = 0

    for niter1 in range(1, max_cycle_outer + 1):
        for (k, kpt) in kpts.loop(1):
            se[k], opt = minimize_chempot(se[k], fock[k], nelec[k], x0=se[k].chempot, **opts)

        for niter2 in range(1, max_cycle_inner + 1):
            nerr = [0] * len(kpts)

            for (k, kpt) in kpts.loop(1):
                w, v = se[k].eig(fock[k], chempot=0.0, out=buf)
                se[k].chempot, nerr[k] = binsearch_chempot((w, v), nmo, nelec[k])

                w, v = se[k].eig(fock[k], out=buf)
                gf[k] = gf[k].__class__(w, v[:nmo], chempot=se[k].chempot)

            rdm1 = gf_to_dm(gf)
            fock = get_fock(Lpq, rdm1, h1e, kpts)
            fock = diis.update(fock, xerr=None)

            nerr = nerr[np.argmax(np.abs(nerr))]
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
        "fock converged = %s  chempot (Γ) = %.9g  dN = %.3g  |ddm| = %.3g",
        converged,
        se[0].chempot,
        nerr,
        derr,
    )

    return gf, se, converged
