"""
Spin-restricted quasiparticle self-consistent GW via self-energy moment
constraints for molecular systems.
"""

import numpy as np
from pyscf import lib

from momentGW import logging, mpi_helper, util
from momentGW.base import BaseGW
from momentGW.evgw import evGW
from momentGW.gw import GW


def kernel(
    gw,
    nmom_max,
    mo_energy,
    mo_coeff,
    moments=None,
    integrals=None,
):
    """
    Moment-constrained quasiparticle self-consistent GW.

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

    if gw.polarizability.lower() == "drpa-exact":
        raise NotImplementedError("%s for polarizability=%s" % (gw.name, gw.polarizability))

    if integrals is None:
        integrals = gw.ao2mo()

    mo_energy = mo_energy.copy()
    mo_coeff = mo_coeff.copy()
    mo_coeff_ref = mo_coeff.copy()

    with util.SilentSCF(gw._scf):
        # Get the overlap
        ovlp = gw._scf.get_ovlp()
        sc = util.einsum("...pq,...qi->...pi", ovlp, mo_coeff)

        # Get the density matrix
        dm = gw._scf.make_rdm1(mo_coeff, gw.mo_occ)
        dm = util.einsum("...pq,...pi,...qj->...ij", dm, np.conj(sc), sc)

        # Get the core Hamiltonian
        h1e = gw._scf.get_hcore()
        h1e = util.einsum("...pq,...pi,...qj->...ij", h1e, np.conj(mo_coeff), mo_coeff)

    diis = util.DIIS()
    diis.space = gw.diis_space

    # Get the self-energy
    solver_options = {} if not gw.solver_options else gw.solver_options.copy()
    for key in gw.solver._opts:
        solver_options[key] = solver_options.get(key, getattr(gw, key, getattr(gw.solver, key)))
    with logging.with_silent():
        subgw = gw.solver(gw._scf, mo_energy=mo_energy, mo_coeff=mo_coeff, **solver_options)
    subconv, gf, se, _ = subgw._kernel(nmom_max, mo_energy, mo_coeff, integrals=integrals)
    logging.write("")
    se_qp = None

    # Get the moments
    th, tp = gw.self_energy_to_moments(se, nmom_max)

    conv = False
    for cycle in range(1, gw.max_cycle + 1):
        with logging.with_comment(f"Start of iteration {cycle}"):
            logging.write("")

        with logging.with_status(f"Iteration {cycle}"):
            # Build the static potential
            se_qp_prev = se_qp if cycle > 1 else None
            se_qp = gw.build_static_potential(mo_energy, se)
            se_qp = diis.update(se_qp)
            if gw.damping != 0.0 and cycle > 1:
                se_qp = (1.0 - gw.damping) * se_qp + gw.damping * se_qp_prev

        # Update the MO energies and orbitals - essentially a Fock
        # loop using the folded static self-energy.
        conv_qp = False
        diis_qp = util.DIIS()
        diis_qp.space = gw.diis_space_qp
        mo_energy_prev = mo_energy.copy()
        with logging.with_table(title="Quasiparticle loop") as table:
            table.add_column("Iter", justify="right")
            table.add_column("Δ (density)", justify="right")

            for qp_cycle in range(1, gw.max_cycle_qp + 1):
                with logging.with_status(f"Iteration [{cycle}, {qp_cycle}]"):
                    fock = integrals.get_fock(dm, h1e)
                    fock_eff = fock + se_qp
                    fock_eff = diis_qp.update(fock_eff)
                    fock_eff = mpi_helper.bcast(fock_eff, root=0)

                    mo_energy, u = np.linalg.eigh(fock_eff)
                    u = mpi_helper.bcast(u, root=0)
                    mo_coeff = util.einsum("...pq,...qi->...pi", mo_coeff_ref, u)

                    dm_prev = dm
                    dm = gw._scf.make_rdm1(u, gw.mo_occ)
                    error = np.max(np.abs(dm - dm_prev))
                    conv_qp = error < gw.conv_tol_qp
                    if qp_cycle in {1, 5, 10, 50, 100, gw.max_cycle_qp} or conv_qp:
                        style = logging.rate(error, gw.conv_tol_qp, gw.conv_tol_qp * 1e2)
                        table.add_row(f"{qp_cycle}", f"[{style}]{error:.3g}[/]")
                    if conv_qp:
                        break

        logging.write(table)
        logging.write("")

        with logging.with_status(f"Iteration {cycle}"):
            # Update the self-energy
            subgw.mo_energy = mo_energy  # FIXME
            subgw.mo_coeff = mo_coeff  # FIXME
            subconv, gf, se, _ = subgw._kernel(nmom_max, mo_energy, mo_coeff)
            gf = gw.project_basis(gf, ovlp, mo_coeff, mo_coeff_ref)
            se = gw.project_basis(se, ovlp, mo_coeff, mo_coeff_ref)

            # Update the moments
            th_prev, tp_prev = th, tp
            th, tp = gw.self_energy_to_moments(se, nmom_max)

            # Check for convergence
            conv = gw.check_convergence(mo_energy, mo_energy_prev, th, th_prev, tp, tp_prev)
            th_prev = th.copy()
            tp_prev = tp.copy()
            with logging.with_comment(f"End of iteration {cycle}"):
                logging.write("")
            if conv:
                break

    return conv, gf, se, mo_energy


class qsGW(GW):  # noqa: D101
    __doc__ = BaseGW.__doc__.format(
        description="Spin-restricted quasiparticle self-consistent GW via self-energy moment "
        + "constraints for molecules.",
        extra_parameters="""max_cycle : int, optional
        Maximum number of iterations. Default value is `50`.
    max_cycle_qp : int, optional
        Maximum number of iterations in the quasiparticle equation
        loop. Default value is `50`.
    conv_tol : float, optional
        Convergence threshold in the change in the HOMO and LUMO.
        Default value is `1e-8`.
    conv_tol_moms : float, optional
        Convergence threshold in the change in the moments. Default
        value is `1e-8`.
    conv_tol_qp : float, optional
        Convergence threshold in the change in the density matrix in
        the quasiparticle equation loop. Default value is `1e-8`.
    conv_logical : callable, optional
        Function that takes an iterable of booleans as input indicating
        whether the individual `conv_tol`, `conv_tol_moms`,
        `conv_tol_qp` have been satisfied, respectively, and returns a
        boolean indicating overall convergence. For example, the
        function `all` requires both metrics to be met, and `any`
        requires just one. Default value is `all`.
    diis_space : int, optional
        Size of the DIIS extrapolation space. Default value is `8`.
    diis_space_qp : int, optional
        Size of the DIIS extrapolation space in the quasiparticle
        loop. Default value is `8`.
    damping : float, optional
        Damping parameter. Default value is `0.0`.
    eta : float, optional
        Small value to regularise the self-energy. Default value is
        `1e-1`.
    srg : float, optional
        If non-zero, use the similarity renormalisation group approach
        of Marie and Loos in place of the `eta` regularisation. For
        value recommendations refer to their paper (arXiv:2303.05984).
        Default value is `0.0`.
    solver : BaseGW, optional
        Solver to use to obtain the self-energy. Compatible with any
        `BaseGW`-like class. Default value is `momentGW.gw.GW`.
    solver_options : dict, optional
        Keyword arguments to pass to the solver. Default value is an
        empty `dict`.
    """,
    )

    # --- Extra qsGW options

    max_cycle = 50
    max_cycle_qp = 50
    conv_tol = 1e-8
    conv_tol_moms = 1e-6
    conv_tol_qp = 1e-8
    conv_logical = all
    diis_space = 8
    diis_space_qp = 8
    damping = 0.0
    eta = 1e-1
    srg = 0.0
    solver = GW
    solver_options = None

    _opts = GW._opts + [
        "max_cycle",
        "max_cycle_qp",
        "conv_tol",
        "conv_tol_moms",
        "conv_tol_qp",
        "conv_logical",
        "diis_space",
        "diis_space_qp",
        "damping",
        "eta",
        "srg",
        "solver",
        "solver_options",
    ]

    @property
    def name(self):
        """Method name."""
        polarizability = self.polarizability.upper().replace("DTDA", "dTDA").replace("DRPA", "dRPA")
        return f"{polarizability}-qsGW"

    _kernel = kernel

    @staticmethod
    def project_basis(matrix, ovlp, mo1, mo2):
        """
        Project a matrix from one basis to another.

        Parameters
        ----------
        matrix : numpy.ndarray or dyson.Lehmann
            Matrix to project. Can also be a `dyson.Lehmann` or object, in
            which case the `couplings` attribute is projected.
        ovlp : numpy.ndarray
            Overlap matrix in the shared (AO) basis.
        mo1 : numpy.ndarray
            First basis, rotates from the shared (AO) basis into the
            basis of `matrix`.
        mo2 : numpy.ndarray
            Second basis, rotates from the shared (AO) basis into the
            desired basis of the output.

        Returns
        -------
        projected_matrix : numpy.ndarray or dyson.Lehmann
            Matrix projected into the desired basis.
        """

        proj = util.einsum("...pq,...pi,...qj->...ij", ovlp, mo1, mo2)

        if isinstance(matrix, np.ndarray):
            projected_matrix = util.einsum("...pq,...pi,...qj->...ij", matrix, proj, proj)
        else:
            coupling = util.einsum("...pk,...pi->...ik", matrix.couplings, proj)
            projected_matrix = matrix.copy()
            projected_matrix.couplings = coupling

        return projected_matrix

    @staticmethod
    def self_energy_to_moments(se, nmom_max):
        """
        Return the hole and particle moments for a self-energy.

        Parameters
        ----------
        se : dyson.Lehmann
            Self-energy to compute the moments of.

        Returns
        -------
        th : numpy.ndarray
            Hole moments.
        tp : numpy.ndarray
            Particle moments.
        """
        th = se.occupied().moment(range(nmom_max + 1))
        tp = se.virtual().moment(range(nmom_max + 1))
        return th, tp

    def build_static_potential(self, mo_energy, se):
        """
        Build the static potential approximation to the self-energy.

        Parameters
        ----------
        mo_energy : numpy.ndarray
            Molecular orbital energies.
        se : dyson.Lehmann
            Self-energy to approximate.

        Returns
        -------
        se_qp : numpy.ndarray
            Static potential approximation to the self-energy.
        """

        if self.srg == 0.0:
            eta = np.sign(se.energies) * self.eta * 1.0j
            denom = lib.direct_sum("p-q-q->pq", mo_energy, se.energies, eta)
            se_i = util.einsum("pk,qk,pk->pq", se.couplings, np.conj(se.couplings), 1 / denom)
            se_j = util.einsum("pk,qk,qk->pq", se.couplings, np.conj(se.couplings), 1 / denom)
        else:
            se_i = np.zeros((mo_energy.size, mo_energy.size), dtype=se.dtype)
            se_j = np.zeros((mo_energy.size, mo_energy.size), dtype=se.dtype)
            for k0, k1 in lib.prange(0, se.naux, 120):
                denom = lib.direct_sum("p-k->pk", mo_energy, se.energies[k0:k1])
                d2p = lib.direct_sum("pk,qk->pqk", denom**2, denom**2)
                reg = 1 - np.exp(-d2p * self.srg)
                reg *= lib.direct_sum("pk,qk->pqk", denom, denom)
                reg /= d2p
                v = se.couplings[:, k0:k1]
                se_i += util.einsum("pk,qk,pqk->pq", v, np.conj(v), reg)
                se_j += se_i.T.conj()

        se_ij = 0.5 * (se_i + se_j)

        if not np.iscomplexobj(se.couplings):
            se_ij = se_ij.real
        else:
            se_ij[np.diag_indices_from(se_ij)] = se_ij[np.diag_indices_from(se_ij)].real

        return se_ij

    check_convergence = evGW.check_convergence

    @property
    def has_fock_loop(self):
        """
        Returns a boolean indicating whether the solver requires a Fock
        loop. In qsGW, this is always `True`.
        """
        return True
