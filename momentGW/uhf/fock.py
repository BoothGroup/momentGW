"""
Fock matrix self-consistent loop for unrestricted references.
"""

import numpy as np
from dyson import Lehmann

from momentGW import logging, mpi_helper, util
from momentGW.fock import minimize_chempot, search_chempot


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
    gw : BaseUGW
        GW object.
    gf : tuple of dyson.Lehmann
        Green's function object for each spin channel.
    se : tuple of dyson.Lehmann
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

    with util.SilentSCF(gw._scf):
        h1e = util.einsum("pq,spi,sqj->sij", gw._scf.get_hcore(), gw.mo_coeff, gw.mo_coeff)
    nmo = gw.nmo
    nocc = gw.nocc
    naux = (se[0].naux, se[1].naux)
    nqmo = (nmo[0] + naux[0], nmo[0] + naux[1])
    nelec = nocc

    diis = util.DIIS()
    diis.space = fock_diis_space
    diis.min_space = fock_diis_min_space
    gf_to_dm = lambda gf: np.array([g.occupied().moment(0) for g in gf])
    rdm1 = gf_to_dm(gf)
    fock = integrals.get_fock(rdm1, h1e)
    gf = list(gf)
    se = list(se)

    buf = np.zeros((max(nqmo), max(nqmo)))
    converged = False
    opts = dict(tol=conv_tol_nelec, maxiter=max_cycle_inner, occupancy=1)
    rdm1_prev = rdm1

    with logging.with_table(title="Fock loop") as table:
        table.add_column("Iter", justify="right")
        table.add_column("Cycle", justify="right")
        table.add_column("Error (nα)", justify="right")
        table.add_column("Error (nβ)", justify="right")
        table.add_column("Δ (density)", justify="right")

        for niter1 in range(1, max_cycle_outer + 1):
            se_α, opt = minimize_chempot(se[0], fock[0], nelec[0], x0=se[0].chempot, **opts)
            se_β, opt = minimize_chempot(se[1], fock[1], nelec[1], x0=se[1].chempot, **opts)
            se = [se_α, se_β]

            for niter2 in range(1, max_cycle_inner + 1):
                with logging.with_status(f"Iteration [{niter1}, {niter2}]"):
                    w, v = se[0].diagonalise_matrix(fock[0], chempot=0.0, out=buf)
                    w = mpi_helper.bcast(w, root=0)
                    v = mpi_helper.bcast(v, root=0)
                    se[0].chempot, nerr_α = search_chempot(w, v, nmo[0], nelec[0], occupancy=1)
                    nerr_α = abs(nerr_α)

                    w, v = se[1].diagonalise_matrix(fock[1], chempot=0.0, out=buf)
                    w = mpi_helper.bcast(w, root=0)
                    v = mpi_helper.bcast(v, root=0)
                    se[1].chempot, nerr_β = search_chempot(w, v, nmo[1], nelec[1], occupancy=1)
                    nerr_β = abs(nerr_β)

                    w, v = se[0].diagonalise_matrix(fock[0], out=buf)
                    w = mpi_helper.bcast(w, root=0)
                    v = mpi_helper.bcast(v, root=0)
                    gf[0] = Lehmann(w, v[: nmo[0]], chempot=se[0].chempot)

                    w, v = se[1].diagonalise_matrix(fock[1], out=buf)
                    w = mpi_helper.bcast(w, root=0)
                    v = mpi_helper.bcast(v, root=0)
                    gf[1] = Lehmann(w, v[: nmo[1]], chempot=se[1].chempot)

                    rdm1 = gf_to_dm(gf)
                    fock = integrals.get_fock(rdm1, h1e)
                    fock = diis.update(fock, xerr=None)

                    derr = np.max(np.absolute(rdm1 - rdm1_prev))
                    if niter2 in {1, 5, 10, 50, 100, max_cycle_inner} or derr < conv_tol_rdm1:
                        nerr_α_style = logging.rate(nerr_α, conv_tol_nelec, conv_tol_nelec * 1e2)
                        nerr_β_style = logging.rate(nerr_β, conv_tol_nelec, conv_tol_nelec * 1e2)
                        derr_style = logging.rate(derr, conv_tol_rdm1, conv_tol_rdm1 * 1e2)
                        table.add_row(
                            f"{niter1}",
                            f"{niter2}",
                            f"[{nerr_α_style}]{nerr_α:.3g}[/]",
                            f"[{nerr_β_style}]{nerr_β:.3g}[/]",
                            f"[{derr_style}]{derr:.3g}[/]",
                        )
                    if derr < conv_tol_rdm1:
                        break

                    rdm1_prev = rdm1.copy()

            if derr < conv_tol_rdm1 and (nerr_α + nerr_β) < conv_tol_nelec:
                converged = True
                break

        logging.write(table)

    return tuple(gf), tuple(se), converged
