"""Spin-restricted quasiparticle self-consistent GW via self-energy moment constraints for molecular
systems.
"""

import copy
from collections import OrderedDict

import numpy as np
from pyscf import lib

from momentGW import logging, mpi_helper, util
from momentGW.evgw import evGW
from momentGW.gw import GW


def kernel(
    gw,
    nmom_max,
    moments=None,
    integrals=None,
):
    """Moment-constrained quasiparticle self-consistent GW.

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

    if gw.polarizability.lower() == "drpa-exact":
        raise NotImplementedError("%s for polarizability=%s" % (gw.name, gw.polarizability))

    # Get the integrals
    if integrals is None:
        integrals = gw.ao2mo()

    # Initialise the orbital
    mo_energy = gw.mo_energy.copy()
    mo_coeff = gw.mo_coeff.copy()

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

    # Get the moments
    subconv, gf, se, _ = subgw._kernel(nmom_max, integrals=integrals)
    logging.write("")
    th, tp = gw.self_energy_to_moments(se, nmom_max)

    # Initialise convergence quantities
    conv = False
    se_qp = None

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
                    # Update the Fock matrix
                    fock = integrals.get_fock(dm, h1e)
                    fock_eff = fock + se_qp
                    fock_eff = diis_qp.update(fock_eff)
                    fock_eff = mpi_helper.bcast(fock_eff, root=0)

                    # Update the MOs
                    mo_energy, u = np.linalg.eigh(fock_eff)
                    u = mpi_helper.bcast(u, root=0)
                    mo_coeff = util.einsum("...pq,...qi->...pi", gw.mo_coeff, u)

                    # Update the density matrix
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


class qsGW(GW):
    """Spin-restricted quasiparticle self-consistent GW via self-energy moment constraints for
    molecules.

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
    """

    _defaults = OrderedDict(
        **GW._defaults,
        max_cycle=50,
        max_cycle_qp=50,
        conv_tol=1e-8,
        conv_tol_moms=1e-6,
        conv_tol_qp=1e-8,
        conv_logical=all,
        diis_space=8,
        diis_space_qp=8,
        damping=0.0,
        eta=1e-1,
        srg=0.0,
        solver=GW,
        solver_options={},
    )

    _kernel = kernel

    check_convergence = evGW.check_convergence

    @property
    def name(self):
        """Get the method name."""
        polarizability = self.polarizability.upper().replace("DTDA", "dTDA").replace("DRPA", "dRPA")
        return f"{polarizability}-qsGW"

    @staticmethod
    def project_basis(matrix, ovlp, mo1, mo2):
        """Project a matrix from one basis to another.

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

        # Build the projection matrix
        proj = util.einsum("...pq,...pi,...qj->...ij", ovlp, mo1, mo2)

        # Project the matrix
        if isinstance(matrix, np.ndarray):
            projected_matrix = util.einsum("...pq,...pi,...qj->...ij", matrix, proj, proj)
        else:
            coupling = util.einsum("...pk,...pi->...ik", matrix.couplings, proj)
            projected_matrix = matrix.copy(couplings=coupling)

        return projected_matrix

    @staticmethod
    def self_energy_to_moments(se, nmom_max):
        """Return the hole and particle moments for a self-energy.

        Parameters
        ----------
        se : dyson.Lehmann
            Self-energy to compute the moments of.
        nmom_max : int
            Maximum moment number to calculate.

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
        """Build the static potential approximation to the self-energy.

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

        # Get the static potential
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

        # Find the Hermitian part
        se_ij = 0.5 * (se_i + se_j)

        # Ensure the static potential is Hermitian
        if not np.iscomplexobj(se.couplings):
            se_ij = se_ij.real
        else:
            se_ij[np.diag_indices_from(se_ij)] = se_ij[np.diag_indices_from(se_ij)].real

        return se_ij

    @property
    def has_fock_loop(self):
        """Get a boolean indicating whether the solver requires a Fock loop.

        In qsGW, this is always `True`.
        """
        return True
