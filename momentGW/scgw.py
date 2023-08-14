"""
Spin-restricted self-consistent GW via self-energy moment constraitns
for molecular systems.
"""

import numpy as np
from pyscf import lib
from pyscf.agf2 import GreensFunction, mpi_helper
from pyscf.ao2mo import _ao2mo
from pyscf.lib import logger

from momentGW import util
from momentGW.base import BaseGW
from momentGW.evgw import evGW
from momentGW.ints import Integrals


def kernel(
    gw,
    nmom_max,
    mo_energy,
    mo_coeff,
    moments=None,
    integrals=None,
):
    """
    Moment-constrained self-consistent GW.

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
    integrals : Integrals, optional
        Density-fitted integrals. If None, generate from scratch.
        Default value is `None`.

    Returns
    -------
    conv : bool
        Convergence flag.
    gf : pyscf.agf2.GreensFunction
        Green's function object
    se : pyscf.agf2.SelfEnergy
        Self-energy object
    qp_energy : numpy.ndarray
        Quasiparticle energies. Always None for scGW, returned for
        compatibility with other scGW methods.
    """

    logger.warn(gw, "scGW is untested!")

    if gw.polarizability == "drpa-exact":
        raise NotImplementedError("%s for polarizability=%s" % (gw.name, gw.polarizability))

    nmo = gw.nmo
    nocc = gw.nocc
    naux = gw.with_df.get_naoaux()

    if integrals is None:
        integrals = gw.ao2mo()

    chempot = 0.5 * (mo_energy[nocc - 1] + mo_energy[nocc])
    gf = GreensFunction(mo_energy, np.eye(mo_energy.size), chempot=chempot)
    gf_ref = gf.copy()

    diis = util.DIIS()
    diis.space = gw.diis_space

    # Get the static part of the SE
    se_static = gw.build_se_static(
        integrals,
        mo_energy=mo_energy,
        mo_coeff=mo_coeff,
    )

    conv = False
    th_prev = tp_prev = np.zeros((nmom_max + 1, nmo, nmo))
    for cycle in range(1, gw.max_cycle + 1):
        logger.info(gw, "%s iteration %d", gw.name, cycle)

        if cycle > 1:
            # Rotate ERIs into (MO, QMO) and (QMO occ, QMO vir)
            # TODO reimplement out keyword
            mo_coeff_g = None if gw.g0 else np.dot(mo_coeff, gf.coupling)
            mo_coeff_w = None if gw.w0 else np.dot(mo_coeff, gf.coupling)
            mo_occ_w = (
                None
                if gw.w0
                else np.array([2] * gf.get_occupied().naux + [0] * gf.get_virtual().naux)
            )
            integrals.update_coeffs(mo_coeff_g=mo_coeff_g, mo_coeff_w=mo_coeff_w, mo_occ_w=mo_occ_w)

        # Update the moments of the SE
        if moments is not None and cycle == 1:
            th, tp = moments
        else:
            th, tp = gw.build_se_moments(
                nmom_max,
                integrals,
                mo_energy=(
                    gf.energy if not gw.g0 else gf_ref.energy,
                    gf.energy if not gw.w0 else gf_ref.energy,
                ),
                mo_occ=(
                    gw._gf_to_occ(gf if not gw.g0 else gf_ref),
                    gw._gf_to_occ(gf if not gw.w0 else gf_ref),
                ),
            )

        # Extrapolate the moments
        try:
            th, tp = diis.update_with_scaling(np.array((th, tp)), (2, 3))
        except Exception as e:
            logger.debug(gw, "DIIS step failed at iteration %d: %s", cycle, repr(e))

        # Damp the moments
        if gw.damping != 0.0:
            th = gw.damping * th_prev + (1.0 - gw.damping) * th
            tp = gw.damping * tp_prev + (1.0 - gw.damping) * tp

        # Solve the Dyson equation
        gf_prev = gf.copy()
        gf, se = gw.solve_dyson(th, tp, se_static, integrals=integrals)

        # Check for convergence
        error_homo = abs(
            gf.energy[np.argmax(gf.coupling[nocc - 1] ** 2)]
            - gf_prev.energy[np.argmax(gf_prev.coupling[nocc - 1] ** 2)]
        )
        error_lumo = abs(
            gf.energy[np.argmax(gf.coupling[nocc] ** 2)]
            - gf_prev.energy[np.argmax(gf_prev.coupling[nocc] ** 2)]
        )
        error_th = gw._moment_error(th, th_prev)
        error_tp = gw._moment_error(tp, tp_prev)
        th_prev = th.copy()
        tp_prev = tp.copy()
        logger.info(gw, "Change in QPs: HOMO = %.6g  LUMO = %.6g", error_homo, error_lumo)
        logger.info(gw, "Change in moments: occ = %.6g  vir = %.6g", error_th, error_tp)
        if gw.conv_logical(
            (
                max(error_homo, error_lumo) < gw.conv_tol,
                max(error_th, error_tp) < gw.conv_tol_moms,
            )
        ):
            conv = True
            break

    return conv, gf, se, None


class scGW(evGW):
    __doc__ = BaseGW.__doc__.format(
        description="Spin-restricted self-consistent GW via self-energy moment constraints for molecules.",
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
    damping : float, optional
        Damping parameter.  Default value is 0.0.
    """,
    )

    @property
    def name(self):
        return "scG%sW%s" % ("0" if self.g0 else "", "0" if self.w0 else "")

    _kernel = kernel
