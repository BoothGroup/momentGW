"""
Spin-restricted eigenvalue self-consistent GW via self-energy moment
constraints for molecular systems.
"""

import numpy as np
from pyscf import lib
from pyscf.lib import logger

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
    Moment-constrained eigenvalue self-consistent GW.

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

    logger.warn(gw, "evGW is untested!")

    if gw.polarizability not in {"drpa"}:
        raise NotImplementedError("%s for polarizability=%s" % (gw.name, gw.polarizability))

    if Lpq is None:
        Lpq = gw.ao2mo(mo_coeff)

    nmo = gw.nmo
    nocc = gw.nocc
    mo_energy = mo_energy.copy()
    mo_energy_ref = mo_energy.copy()

    diis = lib.diis.DIIS()
    diis.space = gw.diis_space

    # Get the static part of the SE
    se_static = gw.build_se_static(
        Lpq=Lpq,
        mo_energy=mo_energy,
        mo_coeff=mo_coeff,
    )

    conv = False
    th_prev = tp_prev = np.zeros((nmom_max + 1, nmo, nmo))
    for cycle in range(1, gw.max_cycle + 1):
        logger.info(gw, "%s iteration %d", gw.name, cycle)

        # Update the moments of the SE
        if moments is not None and cycle == 1:
            th, tp = moments
        else:
            th, tp = gw.build_se_moments(
                nmom_max,
                Lpq=Lpq,
                mo_energy=(
                    mo_energy if not gw.g0 else mo_energy_ref,
                    mo_energy if not gw.w0 else mo_energy_ref,
                ),
            )

        # Extrapolate the moments
        try:
            th, tp = diis.update(np.array((th, tp)))
        except:
            logger.debug("DIIS step failed at iteration %d", cycle)

        # Solve the Dyson equation
        gf, se = gw.solve_dyson(th, tp, se_static, Lpq=Lpq)

        # Update the MO energies
        check = set()
        mo_energy_prev = mo_energy.copy()
        for i in range(nmo):
            arg = np.argmax(gf.coupling[i] ** 2)
            mo_energy[i] = gf.energy[arg]
            check.add(arg)
        if len(check) != nmo:
            logger.warn(gw, "Inconsistent quasiparticle weights!")

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


class evGW(GW):
    __doc__ = BaseGW.__doc__.format(
        description="Spin-restricted eigenvalue self-consistent GW via self-energy moment constraints for molecules.",
        extra_parameters="""g0 : bool, optional
        If `True`, do not self-consistently update the eigenvalues in
        the Green's function.  Default value is `False`.
    w0 : bool, optional
        If `True`, do not self-consistently update the eigenvalues in
        the screened Coulomb interaction.  Default value is `False`.
    max_cycle : int, optional
        Maximum number of iterations.  Default value is 50.
    conv_tol : float, optional
        Convergence threshold in the change in the HOMO and LUMO.
        Default value is 1e-8.
    conv_tol_moms : float, optional
        Convergence threshold in the change in the moments. Default
        value is 1e-8.
    diis_space : int, optional
        Size of the DIIS extrapolation space.  Default value is 8.
    """,
    )

    # --- Extra evGW options

    g0 = False
    w0 = False
    max_cycle = 50
    conv_tol = 1e-8
    conv_tol_moms = 1e-6
    diis_space = 8

    _opts = GW._opts + ["g0", "w0", "max_cycle", "conv_tol", "conv_tol_moms", "diis_space"]

    @property
    def name(self):
        return "evG%sW%s" % ("0" if self.g0 else "", "0" if self.w0 else "")

    def kernel(
        self,
        nmom_max,
        mo_energy=None,
        mo_coeff=None,
        moments=None,
        Lpq=None,
    ):
        if mo_coeff is None:
            mo_coeff = self.mo_coeff
        if mo_energy is None:
            mo_energy = self.mo_energy

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
