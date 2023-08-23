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
        Default value is `None`.
    integrals : Integrals, optional
        Integrals object. If `None`, generate from scratch. Default
        value is `None`.

    Returns
    -------
    conv : bool
        Convergence flag.
    gf : pyscf.agf2.GreensFunction
        Green's function object
    se : pyscf.agf2.SelfEnergy
        Self-energy object
    qp_energy : numpy.ndarray
        Quasiparticle energies. Always `None` for scGW, returned for
        compatibility with other scGW methods.
    """

    if gw.polarizability == "drpa-exact":
        raise NotImplementedError("%s for polarizability=%s" % (gw.name, gw.polarizability))

    if integrals is None:
        integrals = gw.ao2mo()

    gf_ref = gf = gw.init_gf(mo_energy)

    diis = util.DIIS()
    diis.space = gw.diis_space

    # Get the static part of the SE
    se_static = gw.build_se_static(
        integrals,
        mo_energy=mo_energy,
        mo_coeff=mo_coeff,
    )

    conv = False
    th_prev = tp_prev = None
    for cycle in range(1, gw.max_cycle + 1):
        logger.info(gw, "%s iteration %d", gw.name, cycle)

        if cycle > 1:
            # Rotate ERIs into (MO, QMO) and (QMO occ, QMO vir)
            integrals.update_coeffs(
                mo_coeff_g=(
                    None
                    if gw.g0
                    else lib.einsum("...pq,...qi->...pi", mo_coeff, gw._gf_to_coupling(gf))
                ),
                mo_coeff_w=(
                    None
                    if gw.w0
                    else lib.einsum("...pq,...qi->...pi", mo_coeff, gw._gf_to_coupling(gf))
                ),
                mo_occ_w=None if gw.w0 else gw._gf_to_occ(gf),
            )

        # Update the moments of the SE
        if moments is not None and cycle == 1:
            th, tp = moments
        else:
            th, tp = gw.build_se_moments(
                nmom_max,
                integrals,
                mo_energy=(
                    gw._gf_to_energy(gf if not gw.g0 else gf_ref),
                    gw._gf_to_energy(gf if not gw.w0 else gf_ref),
                ),
                mo_occ=(
                    gw._gf_to_occ(gf if not gw.g0 else gf_ref),
                    gw._gf_to_occ(gf if not gw.w0 else gf_ref),
                ),
            )

        # Extrapolate the moments
        try:
            th, tp = diis.update_with_scaling(np.array((th, tp)), (-2, -1))
        except Exception as e:
            logger.debug(gw, "DIIS step failed at iteration %d", cycle)

        # Damp the moments
        if gw.damping != 0.0 and cycle > 1:
            th = gw.damping * th_prev + (1.0 - gw.damping) * th
            tp = gw.damping * tp_prev + (1.0 - gw.damping) * tp

        # Solve the Dyson equation
        gf, se = gw.solve_dyson(th, tp, se_static, integrals=integrals)

        # Update the MO energies
        mo_energy_prev = mo_energy.copy()
        mo_energy = gw._gf_to_mo_energy(gf)

        # Check for convergence
        conv = gw.check_convergence(mo_energy, mo_energy_prev, th, th_prev, tp, tp_prev)
        th_prev = th.copy()
        tp_prev = tp.copy()
        if conv:
            break

    return conv, gf, se, None


class scGW(evGW):
    __doc__ = BaseGW.__doc__.format(
        description="Spin-restricted self-consistent GW via self-energy moment constraints for molecules.",
        extra_parameters="""g0 : bool, optional
        If `True`, do not self-consistently update the eigenvalues in
        the Green's function. Default value is `False`.
    w0 : bool, optional
        If `True`, do not self-consistently update the eigenvalues in
        the screened Coulomb interaction. Default value is `False`.
    max_cycle : int, optional
        Maximum number of iterations. Default value is `50`.
    conv_tol : float, optional
        Convergence threshold in the change in the HOMO and LUMO.
        Default value is `1e-8`.
    conv_tol_moms : float, optional
        Convergence threshold in the change in the moments. Default
        value is `1e-8`.
    diis_space : int, optional
        Size of the DIIS extrapolation space. Default value is `8`.
    damping : float, optional
        Damping parameter. Default value is `0.0`.
    """,
    )

    @property
    def name(self):
        return "scG%sW%s" % ("0" if self.g0 else "", "0" if self.w0 else "")

    _kernel = kernel
