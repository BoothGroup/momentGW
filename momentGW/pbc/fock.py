"""
Fock matrix and static self-energy parts with periodic boundary
conditions.
"""

import numpy as np
import scipy.optimize
from dyson import Lehmann
from pyscf import lib

from momentGW import logging, mpi_helper, util
from momentGW.fock import ChemicalPotentialError


# TODO inherit
def _gradient(x, se, fock, nelec, occupancy=2, buf=None):
    """Gradient of the number of electrons w.r.t shift in auxiliary
    energies.
    """
    # TODO buf

    ws, vs = zip(*[s.diagonalise_matrix(f, chempot=x) for s, f in zip(se, fock)])
    chempot, error = search_chempot(ws, vs, se[0].nphys, nelec, occupancy=occupancy)

    nmo = se[0].nphys

    ddm = 0.0
    for i in mpi_helper.nrange(len(ws)):
        w, v = ws[i], vs[i]
        nmo = se[i].nphys
        nocc = np.sum(w < chempot)
        h1 = -np.dot(v[nmo:, nocc:].T.conj(), v[nmo:, :nocc])
        zai = -h1 / lib.direct_sum("i-a->ai", w[:nocc], w[nocc:])
        ddm += util.einsum("ai,pa,pi->", zai, v[:nmo, nocc:], v[:nmo, :nocc].conj()).real * 4

    ddm = mpi_helper.allreduce(ddm)
    grad = occupancy * error * ddm

    return error**2, grad


def search_chempot_constrained(w, v, nphys, nelec, occupancy=2):
    """
    Search for a chemical potential, constraining the k-point
    dependent occupancy to ensure no crossover of states. If this
    is not possible, a ValueError will be raised.
    """

    if nelec == 0:
        return min(wk[0] for wk in w) - 1e-6, 0.0

    nmo = max(len(x) for x in w)
    nkpts = len(w)
    sum0 = sum1 = 0.0

    for i in range(nmo):
        n = 0
        for k in range(nkpts):
            n += np.dot(v[k][:nphys, i].conj().T, v[k][:nphys, i]).real
        n *= occupancy
        sum0, sum1 = sum1, sum1 + n

        if i > 0 and sum0 <= nelec and nelec <= sum1:
            break

    if abs(sum0 - nelec) < abs(sum1 - nelec):
        homo = i - 1
        error = nelec - sum0
    else:
        homo = i
        error = nelec - sum1

    lumo = homo + 1

    if lumo == nmo:
        chempot = np.max(w) + 1e-6
    else:
        e_homo = np.max([x[homo] for x in w])
        e_lumo = np.min([x[lumo] for x in w])

        if e_homo > e_lumo:
            raise ChemicalPotentialError(
                "Could not find a chemical potential under "
                "the constrain of equal k-point occupancy."
            )

        chempot = 0.5 * (e_homo + e_lumo)

    return chempot, error


def search_chempot_unconstrained(w, v, nphys, nelec, occupancy=2):
    """
    Search for a chemical potential, without constraining the
    k-point dependent occupancy.
    """

    w = np.concatenate(w)
    v = np.hstack([vk[:nphys] for vk in v])

    mask = np.argsort(w)
    w = w[mask]
    v = v[:, mask]

    nmo = v.shape[-1]
    sum0 = sum1 = 0.0

    for i in range(nmo):
        n = occupancy * np.dot(v[:nphys, i].conj().T, v[:nphys, i]).real
        sum0, sum1 = sum1, sum1 + n

        if i > 0 and sum0 <= nelec and nelec <= sum1:
            break

    if abs(sum0 - nelec) < abs(sum1 - nelec):
        homo = i - 1
        error = nelec - sum0
    else:
        homo = i
        error = nelec - sum1

    lumo = homo + 1

    if lumo == len(w):
        chempot = w[homo] + 1e-6
    else:
        chempot = 0.5 * (w[homo] + w[lumo])

    return chempot, error


def search_chempot(w, v, nphys, nelec, occupancy=2):
    """
    Search for a chemical potential, first trying with k-point
    restraints and if that doesn't succeed then without.
    """

    try:
        chempot, error = search_chempot_constrained(w, v, nphys, nelec, occupancy=occupancy)
    except ChemicalPotentialError:
        chempot, error = search_chempot_unconstrained(w, v, nphys, nelec, occupancy=occupancy)

    return chempot, error


@logging.with_timer("Chemical potential optimisation")
@logging.with_status("Optimising chemical potential")
def minimize_chempot(se, fock, nelec, occupancy=2, x0=0.0, tol=1e-6, maxiter=200):
    """
    Optimise the shift in auxiliary energies to satisfy the electron
    number, ensuring that the same shift is applied at all k-points.
    """

    tol = tol**2  # we minimize the squared error
    dtype = np.result_type(*[s.dtype for s in se], *[f.dtype for f in fock])
    nphys = max([s.nphys for s in se])
    naux = max([s.naux for s in se])
    buf = np.zeros(((nphys + naux) ** 2,), dtype=dtype)
    fargs = (se, fock, nelec, occupancy, buf)

    options = dict(maxiter=maxiter, ftol=tol, xtol=tol, gtol=tol)
    kwargs = dict(x0=x0, method="TNC", jac=True, options=options)
    fun = _gradient

    opt = scipy.optimize.minimize(fun, args=fargs, **kwargs)

    for s in se:
        s.energies -= opt.x

    ws, vs = zip(*[s.diagonalise_matrix(f) for s, f in zip(se, fock)])
    chempot = search_chempot(ws, vs, se[0].nphys, nelec, occupancy=occupancy)[0]

    for s in se:
        s.chempot = chempot

    return se, opt


@logging.with_timer("Fock loop")
@logging.with_status("Running Fock loop")
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
    gw : BaseKGW
        GW object.
    gf : tuple of dyson.Lehmann
        Green's function object at each k-point.
    se : tuple of dyson.Lehmann
        Self-energy object at each k-point.
    integrals : KIntegrals, optional
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

    with util.SilentSCF(gw._scf):
        h1e = util.einsum(
            "kpq,kpi,kqj->kij",
            gw._scf.get_hcore(),
            np.conj(gw.mo_coeff),
            gw.mo_coeff,
        )
    nmo = gw.nmo
    nocc = gw.nocc
    naux = [s.naux for s in se]
    nqmo = [nmo + n for n in naux]
    nelec = [n * 2 for n in nocc]
    kpts = gw.kpts

    diis = util.DIIS()
    diis.space = fock_diis_space
    diis.min_space = fock_diis_min_space
    gf_to_dm = lambda gf: np.array([g.occupied().moment(0) for g in gf]) * 2.0
    rdm1 = gf_to_dm(gf)
    fock = integrals.get_fock(rdm1, h1e)

    buf = np.zeros((max(nqmo), max(nqmo)), dtype=complex)
    converged = False
    opts = dict(tol=conv_tol_nelec, maxiter=max_cycle_inner)
    rdm1_prev = rdm1

    with logging.with_table(title="Fock loop") as table:
        table.add_column("Iter", justify="right")
        table.add_column("Cycle", justify="right")
        table.add_column("Error (nelec)", justify="right")
        table.add_column("Δ (density)", justify="right")

        for niter1 in range(1, max_cycle_outer + 1):
            se, opt = minimize_chempot(se, fock, sum(nelec), x0=se[0].chempot, **opts)

            for niter2 in range(1, max_cycle_inner + 1):
                with logging.with_status(f"Iteration [{niter1}, {niter2}]"):
                    w, v = zip(
                        *[s.diagonalise_matrix(f, chempot=0.0, out=buf) for s, f in zip(se, fock)]
                    )
                    w = [mpi_helper.bcast(wk, root=0) for wk in w]
                    v = [mpi_helper.bcast(vk, root=0) for vk in v]
                    chempot, nerr = search_chempot(w, v, nmo, sum(nelec))
                    nerr = abs(nerr)

                    for k in kpts.loop(1):
                        se[k].chempot = chempot
                        w, v = se[k].diagonalise_matrix(fock[k], out=buf)
                        gf[k] = Lehmann(w, v[:nmo], chempot=se[k].chempot)

                    rdm1 = gf_to_dm(gf)
                    fock = integrals.get_fock(rdm1, h1e)
                    fock = diis.update(fock, xerr=None)

                    derr = np.max(np.absolute(rdm1 - rdm1_prev))
                    if niter2 in {1, 5, 10, 50, 100, max_cycle_inner} or derr < conv_tol_rdm1:
                        nerr_style = logging.rate(nerr, conv_tol_nelec, conv_tol_nelec * 1e2)
                        derr_style = logging.rate(derr, conv_tol_rdm1, conv_tol_rdm1 * 1e2)
                        table.add_row(
                            f"{niter1}",
                            f"{niter2}",
                            f"[{nerr_style}]{nerr:.3g}[/]",
                            f"[{derr_style}]{derr:.3g}[/]",
                        )
                    if derr < conv_tol_rdm1:
                        break

                    rdm1_prev = rdm1.copy()

            if derr < conv_tol_rdm1 and nerr < conv_tol_nelec:
                converged = True
                break

        logging.write(table)

    return gf, se, converged
