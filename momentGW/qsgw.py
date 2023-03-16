"""
Spin-restricted quasiparticle self-consistent GW via self-energy moment
constraints for molecular systems.
"""

import numpy as np
from pyscf import lib
from pyscf.lib import logger
from pyscf.ao2mo import _ao2mo
from pyscf.agf2.dfragf2 import get_jk

from momentGW.base import BaseGW
from momentGW.gw import GW


def kernel(
    gw,
    nmom_max,
    mo_energy,
    mo_coeff,
    moments=None,
    Lpq=None,
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
        Default value is None.
    Lpq : np.ndarray, optional
        Density-fitted ERI tensor. If None, generate from `gw.ao2mo`.
        Default value is None.

    Returns
    -------
    conv : bool
        Convergence flag.
    gf : pyscf.agf2.GreensFunction
        Green's function object
    se : pyscf.agf2.SelfEnergy
        Self-energy object
    """

    logger.warn(gw, "qsGW is untested!")

    if gw.polarizability not in {"drpa"}:
        raise NotImplementedError("%s for polarizability=%s" % (gw.name, gw.polarizability))

    if Lpq is None:
        Lpq = gw.ao2mo(mo_coeff)

    nmo = gw.nmo
    nocc = gw.nocc
    naux = gw.with_df.get_naoaux()
    mo_energy = mo_energy.copy()
    mo_energy_ref = mo_energy.copy()
    mo_coeff = mo_coeff.copy()
    mo_coeff_ref = mo_coeff.copy()

    dm = np.eye(gw.nmo) * 2
    dm[nocc:, nocc:] = 0
    h1e = np.linalg.multi_dot((mo_coeff.T, gw._scf.get_hcore(), mo_coeff))

    diis = lib.diis.DIIS()
    diis.space = gw.diis_space

    ovlp = gw._scf.get_ovlp()
    def project_basis(m, c1, c2):
        # Project m from MO basis c1 to MO basis c2
        p = np.linalg.multi_dot((c1.T, ovlp, c2))
        m = lib.einsum("...pq,pi,qj->...ij", m, p, p)
        return m

    # Get the self-energy
    subgw = gw.solver(gw._scf, **(gw.solver_options if gw.solver_options else {}))
    subgw.verbose = 0
    subgw.mo_energy = mo_energy
    subgw.mo_coeff = mo_coeff
    subconv, gf, se = subgw.kernel(nmom_max=nmom_max)

    # Get the moments
    th = se.get_occupied().moment(range(nmom_max+1))
    tp = se.get_virtual().moment(range(nmom_max+1))

    conv = False
    for cycle in range(1, gw.max_cycle + 1):
        logger.info(gw, "%s iteration %d", gw.name, cycle)

        # Build the static potential
        denom = lib.direct_sum("p-q-q->pq", mo_energy, se.energy, np.sign(se.energy) * 1.0j * gw.eta)
        se_qp = lib.einsum("pk,qk,pk->pq", se.coupling, se.coupling, 1/denom).real
        se_qp = 0.5 * (se_qp + se_qp.T)
        se_qp = project_basis(se_qp, mo_coeff, mo_coeff_ref)
        se_qp = diis.update(se_qp)

        # Update the MO energies and orbitals - essentially a Fock
        # loop using the folded static self-energy.
        conv_qp = False
        diis_qp = lib.diis.DIIS()
        diis_qp.space = gw.diis_space_qp
        mo_energy_prev = mo_energy.copy()
        for qp_cycle in range(1, gw.max_cycle_qp+1):
            j, k = get_jk(gw, lib.pack_tril(Lpq, axis=-1), dm)
            fock_eff = h1e + j - 0.5 * k + se_qp
            fock_eff = diis_qp.update(fock_eff)

            mo_energy, u = np.linalg.eigh(fock_eff)
            mo_coeff = np.dot(mo_coeff_ref, u)

            dm_prev = dm
            dm = np.dot(u[:, :nocc], u[:, :nocc].T) * 2
            error = np.max(np.abs(dm - dm_prev))
            if error < gw.conv_tol_qp:
                conv_qp = True
                break

        if conv_qp:
            logger.info(gw, "QP loop converged.")
        else:
            logger.info(gw, "QP loop failed to converge.")

        # Update the self-energy
        subgw.mo_energy = mo_energy
        subgw.mo_coeff = mo_coeff
        _, gf, se = subgw.kernel(nmom_max=nmom_max)

        # Update the moments
        th_prev, tp_prev = th, tp
        th = se.get_occupied().moment(range(nmom_max+1))
        th = project_basis(th, mo_coeff, mo_coeff_ref)
        tp = se.get_virtual().moment(range(nmom_max+1))
        tp = project_basis(tp, mo_coeff, mo_coeff_ref)

        # Check for convergence
        error_homo = abs(mo_energy[nocc - 1] - mo_energy_prev[nocc - 1])
        error_lumo = abs(mo_energy[nocc] - mo_energy_prev[nocc])
        error_th = gw._moment_error(th, th_prev)
        error_tp = gw._moment_error(tp, tp_prev)
        th_prev = th.copy()
        tp_prev = tp.copy()
        logger.info(gw, "Change in QPs: HOMO = %.6g  LUMO = %.6g", error_homo, error_lumo)
        logger.info(gw, "Change in moments: occ = %.6g  vir = %.6g", error_th, error_tp)
        if max(error_homo, error_lumo) < gw.conv_tol:
            if max(error_th, error_tp) < gw.conv_tol_moms:
                conv = True
                break

    return conv, gf, se


class qsGW(GW):
    __doc__ = BaseGW.__doc__.format(
        description="Spin-restricted quasiparticle self-consistent GW via self-energy moment constraints for molecules.",
        extra_parameters="""max_cycle : int, optional
        Maximum number of iterations.  Default value is 50.
    max_cycle_qp : int, optional
        Maximum number of iterations in the quasiparticle equation
        loop.  Default value is 50.
    conv_tol : float, optional
        Convergence threshold in the change in the HOMO and LUMO.
        Default value is 1e-8.
    conv_tol_moms : float, optional
        Convergence threshold in the change in the moments. Default
        value is 1e-8.
    conv_tol_qp : float, optional
        Convergence threshold in the change in the density matrix in
        the quasiparticle equation loop.  Default value is 1e-8.
    diis_space : int, optional
        Size of the DIIS extrapolation space.  Default value is 8.
    diis_space_qp : int, optional
        Size of the DIIS extrapolation space in the quasiparticle
        loop.  Default value is 8.
    eta : float, optional
        Small value to regularise the self-energy.  Default value is
        `1e-1`.
    solver : BaseGW, optional
        Solver to use to obtain the self-energy.  Compatible with any
        `BaseGW`-like class.  Default value is `momentGW.gw.GW`.
    solver_options : dict, optional
        Keyword arguments to pass to the solver.  Default value is an
        emtpy `dict`.
    """,
    )

    # --- Extra qsGW options

    max_cycle = 50
    max_cycle_qp = 50
    conv_tol = 1e-8
    conv_tol_moms = 1e-6
    conv_tol_qp = 1e-8
    diis_space = 8
    diis_space_qp = 8
    eta = 1e-1
    solver = GW
    solver_options = None

    _opts = GW._opts + ["max_cycle", "max_cycle_qp", "conv_tol", "conv_tol_moms", "conv_tol_qp", "diis_space", "diis_space_qp", "eta", "solver", "solver_options"]

    @property
    def name(self):
        return "qsGW"

    def kernel(
        self,
        nmom_max,
        mo_energy=None,
        mo_coeff=None,
        moments=None,
        Lpq=None,
    ):
        if mo_coeff is None:
            mo_coeff = self._scf.mo_coeff
        if mo_energy is None:
            mo_energy = self._scf.mo_energy

        cput0 = (logger.process_clock(), logger.perf_counter())
        self.dump_flags()
        logger.info(self, "nmom_max = %d", nmom_max)

        self.converged, self.gf, self.se = kernel(
            self,
            nmom_max,
            mo_energy,
            mo_coeff,
            Lpq=Lpq,
        )

        gf_occ = self.gf.get_occupied()
        gf_occ.remove_uncoupled(tol=1e-1)
        for n in range(min(5, gf_occ.naux)):
            en = -gf_occ.energy[-(n + 1)]
            vn = gf_occ.coupling[:, -(n + 1)]
            qpwt = np.linalg.norm(vn) ** 2
            logger.note(self, "IP energy level %d E = %.16g  QP weight = %0.6g", n, en, qpwt)

        gf_vir = self.gf.get_virtual()
        gf_vir.remove_uncoupled(tol=1e-1)
        for n in range(min(5, gf_vir.naux)):
            en = gf_vir.energy[n]
            vn = gf_vir.coupling[:, n]
            qpwt = np.linalg.norm(vn) ** 2
            logger.note(self, "EA energy level %d E = %.16g  QP weight = %0.6g", n, en, qpwt)

        if self.converged:
            logger.note(self, "%s converged", self.name)
        else:
            logger.note(self, "%s failed to converge", self.name)

        logger.timer(self, self.name, *cput0)

        return self.converged, self.gf, self.se
