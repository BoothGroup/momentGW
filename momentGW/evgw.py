"""
Spin-restricted eigenvalue self-consistent GW via self-energy moment
constraints for molecular systems.
"""

import numpy as np
from pyscf import lib
from pyscf.lib import logger

from momentGW import util
from momentGW.base import BaseGW
from momentGW.gw import GW
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
        Quasiparticle energies. Always None for evGW, returned for
        compatibility with other evGW methods.
    """

    if gw.polarizability == "drpa-exact":
        raise NotImplementedError("%s for polarizability=%s" % (gw.name, gw.polarizability))

    if integrals is None:
        integrals = gw.ao2mo()

    mo_energy = mo_energy.copy()
    mo_energy_ref = mo_energy.copy()

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

        # Update the moments of the SE
        if moments is not None and cycle == 1:
            th, tp = moments
        else:
            th, tp = gw.build_se_moments(
                nmom_max,
                integrals,
                mo_energy=(
                    mo_energy if not gw.g0 else mo_energy_ref,
                    mo_energy if not gw.w0 else mo_energy_ref,
                ),
            )

        # Extrapolate the moments
        try:
            th, tp = diis.update_with_scaling(np.array((th, tp)), (-2, -1))
        except:
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
        Maximum number of iterations.  Default value is `50`.
    conv_tol : float, optional
        Convergence threshold in the change in the HOMO and LUMO.
        Default value is `1e-8`.
    conv_tol_moms : float, optional
        Convergence threshold in the change in the moments. Default
        value is `1e-8`.
    conv_logical : callable, optional
        Function that takes an iterable of booleans as input indicating
        whether the individual `conv_tol` and `conv_tol_moms` have been
        satisfied, respectively, and returns a boolean indicating
        overall convergence. For example, the function `all` requires
        both metrics to be met, and `any` requires just one. Default
        value is `all`.
    diis_space : int, optional
        Size of the DIIS extrapolation space.  Default value is `8`.
    damping : float, optional
        Damping parameter.  Default value is `0.0`.
    """,
    )

    # --- Extra evGW options

    g0 = False
    w0 = False
    max_cycle = 50
    conv_tol = 1e-8
    conv_tol_moms = 1e-6
    conv_logical = all
    diis_space = 8
    damping = 0.0

    _opts = GW._opts + [
        "g0",
        "w0",
        "max_cycle",
        "conv_tol",
        "conv_tol_moms",
        "conv_logical",
        "diis_space",
        "damping",
    ]

    @property
    def name(self):
        return "evG%sW%s" % ("0" if self.g0 else "", "0" if self.w0 else "")

    _kernel = kernel

    def check_convergence(self, mo_energy, mo_energy_prev, th, th_prev, tp, tp_prev):
        """Check for convergence, and print a summary of changes.

        Parameters
        ----------
        mo_energy : numpy.ndarray
            Molecular orbital energies.
        mo_energy_prev : numpy.ndarray
            Molecular orbital energies from the previous iteration.
        th : numpy.ndarray
            Moments of the occupied self-energy.
        th_prev : numpy.ndarray
            Moments of the occupied self-energy from the previous iteration.
        tp : numpy.ndarray
            Moments of the virtual self-energy.
        tp_prev : numpy.ndarray
            Moments of the virtual self-energy from the previous iteration.

        Returns
        -------
        conv : bool
            Convergence flag.
        """

        if th_prev is None:
            th_prev = np.zeros_like(th)
        if tp_prev is None:
            tp_prev = np.zeros_like(tp)

        error_homo = abs(mo_energy[self.nocc - 1] - mo_energy_prev[self.nocc - 1])
        error_lumo = abs(mo_energy[self.nocc] - mo_energy_prev[self.nocc])

        error_th = self._moment_error(th, th_prev)
        error_tp = self._moment_error(tp, tp_prev)

        logger.info(self, "Change in QPs: HOMO = %.6g  LUMO = %.6g", error_homo, error_lumo)
        logger.info(self, "Change in moments: occ = %.6g  vir = %.6g", error_th, error_tp)

        return self.conv_logical(
            (
                max(error_homo, error_lumo) < self.conv_tol,
                max(error_th, error_tp) < self.conv_tol_moms,
            )
        )
