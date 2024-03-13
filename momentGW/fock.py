"""
Fock matrix self-consistent loop.
"""

import numpy as np
import scipy
from dyson import Lehmann
from pyscf import lib

from momentGW import logging, mpi_helper, util


class ChemicalPotentialError(ValueError):
    """Exception raised when the chemical potential cannot be found."""

    pass


def _gradient(x, se, fock, nelec, occupancy=2, buf=None):
    """Gradient of the number of electrons w.r.t shift in auxiliary
    energies.
    """
    # TODO buf

    w, v = se.diagonalise_matrix(fock, chempot=x)
    chempot, error = search_chempot(w, v, se.nphys, nelec, occupancy=occupancy)

    nocc = np.sum(w < chempot)
    nmo = se.nphys

    h1 = -np.dot(v[nmo:, nocc:].T.conj(), v[nmo:, :nocc])
    zai = -h1 / lib.direct_sum("i-a->ai", w[:nocc], w[nocc:])
    ddm = util.einsum("ai,pa,pi->", zai, v[:nmo, nocc:], v[:nmo, :nocc].conj()).real * 4
    grad = occupancy * error * ddm

    return error**2, grad


def search_chempot(w, v, nphys, nelec, occupancy=2):
    """
    Search for a chemical potential.
    """

    if nelec == 0:
        return w[0] - 1e-6, 0.0

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


@logging.with_timer("Chemical potential optimisation")
@logging.with_status("Optimising chemical potential")
def minimize_chempot(se, fock, nelec, occupancy=2, x0=0.0, tol=1e-6, maxiter=200):
    """
    Optimise the shift in auxiliary energies to satisfy the electron
    number.
    """

    tol = tol**2  # we minimize the squared error
    dtype = np.result_type(se.dtype, fock.dtype)
    nphys = se.nphys
    naux = se.naux
    buf = np.zeros(((nphys + naux) ** 2,), dtype=dtype)
    fargs = (se, fock, nelec, occupancy, buf)

    options = dict(maxiter=maxiter, ftol=tol, xtol=tol, gtol=tol)
    kwargs = dict(x0=x0, method="TNC", jac=True, options=options)
    fun = _gradient

    opt = scipy.optimize.minimize(fun, args=fargs, **kwargs)

    se.energies -= opt.x
    w, v = se.diagonalise_matrix(fock)
    se.chempot = search_chempot(w, v, se.nphys, nelec, occupancy=occupancy)[0]

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
    gw : BaseGW
        GW object.
    gf : dyson.Lehmann
        Green's function object.
    se : dyson.Lehmann
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
    gf_to_dm = lambda gf: gf.occupied().moment(0) * 2.0
    rdm1 = gf_to_dm(gf)
    fock = integrals.get_fock(rdm1, h1e)

    buf = np.zeros((nqmo, nqmo))
    converged = False
    opts = dict(tol=conv_tol_nelec, maxiter=max_cycle_inner)
    rdm1_prev = 0

    with logging.with_table(title="Fock loop") as table:
        table.add_column("Iter", justify="right")
        table.add_column("Cycles", justify="right")
        table.add_column("Error (nelec)", justify="right")
        table.add_column("Δ (density)", justify="right")

        for niter1 in range(1, max_cycle_outer + 1):
            se, opt = minimize_chempot(se, fock, nelec, x0=se.chempot, **opts)

            for niter2 in range(1, max_cycle_inner + 1):
                with logging.with_status(f"Iteration [{niter1}, {niter2}]"):
                    w, v = se.diagonalise_matrix(fock, chempot=0.0, out=buf)
                    w = mpi_helper.bcast(w, root=0)
                    v = mpi_helper.bcast(v, root=0)
                    se.chempot, nerr = search_chempot(w, v, nmo, nelec)

                    w, v = se.diagonalise_matrix(fock, out=buf)
                    w = mpi_helper.bcast(w, root=0)
                    v = mpi_helper.bcast(v, root=0)
                    gf = Lehmann(w, v[:nmo], chempot=se.chempot)

                    rdm1 = gf_to_dm(gf)
                    fock = integrals.get_fock(rdm1, h1e)
                    fock = diis.update(fock, xerr=None)

                    if niter2 > 1:
                        derr = np.max(np.absolute(rdm1 - rdm1_prev))
                        if derr < conv_tol_rdm1:
                            break

                    rdm1_prev = rdm1.copy()

            nerr_style = logging.rate(abs(nerr), conv_tol_nelec, conv_tol_nelec * 1e2)
            derr_style = logging.rate(derr, conv_tol_rdm1, conv_tol_rdm1 * 1e2)
            table.add_row(
                f"{niter1}",
                f"{niter2}",
                f"[{nerr_style}]{nerr:.3g}[/]",
                f"[{derr_style}]{derr:.3g}[/]",
            )

            if derr < conv_tol_rdm1 and abs(nerr) < conv_tol_nelec:
                converged = True
                break

        logging.write(table)

    return gf, se, converged
