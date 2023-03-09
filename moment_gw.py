"""
Spin-restricted G0W0 via self-energy moment constraints

This implementation should be N^4, if the screened Coulomb moments
are computed via numerical integration, or N^6 otherwise. It should
converge to exact full-frequency integration as the number of self-energy
moments computed increases.

Note that the 'quasi-particle' equation is solved exactly via full
Dyson inversion in N^3 time, without diagonal self-energy approximation,
and will return all possible Greens function poles (which may be more
than the original mean-field), along with their quasi-particle weights.
This avoids the traditional diagonal self-energy approximation and assumption
that self-energy poles are far from MO energies.
"""

from functools import reduce, partial
import scipy.special
import numpy
import numpy as np
import h5py

from pyscf import lib
from pyscf.lib import logger
from pyscf.ao2mo import _ao2mo
from pyscf import df, scf
from pyscf.mp.mp2 import get_nocc, get_nmo, get_frozen_mask
from pyscf.agf2.aux_space import GreensFunction, SelfEnergy, combine
from pyscf.agf2 import chempot
from pyscf import __config__
import rpamoms

einsum = lib.einsum

DEBUG = True

# TODO:
# Make nmom notation consistent with AGF2
# orbs argument
# frozen argument


def _kernel(
    agw,
    nmom,
    mo_energy,
    mo_coeff,
    moments=None,
    Lpq=None,
    orbs=None,
    vhf_df=False,
    npoints=48,
    verbose=logger.NOTE,
):
    """Moment-constrained G0W0. Returns the Green's function and self-
    energy as objects using the `GreensFunction` and `SelfEnergy`
    objects from the `agf2` module.

    Parameters
    ----------
    agw : AGW
        AGW object.
    nmom : int
        Maximum moment number to calculate.
    mo_energy : numpy.ndarray
        Molecular orbital energies.
    mo_coeff : numpy.ndarray
        Molecular orbital coefficients.
    moments : tuple of numpy.ndarray, optional
        Tuple of (hole, particle) moments, if passed then they will
        be used instead of calculating them. Default value is None.
    Lpq : np.ndarray, optional
        Density-fitted ERI tensor. If None, generate from `agw.ao2mo`.
        Default value is None.
    orbs : list of int, optional
        List of orbitals to include in GW calculation. If None,
        include all orbitals. Default value is None.
    vhf_df : bool, optional
        If True, calculate the static self-energy directly from `Lpq`.
        Default value is False.

    Returns
    -------
    conv : bool
        Convergence flag. Always True for AGW, returned for
        compatibility with other GW methods.
    gf : pyscf.agf2.GreensFunction
        Green's function object
    se : pyscf.agf2.SelfEnergy
        Self-energy object
    """

    mf = agw._scf
    if agw.frozen is None:
        frozen = 0
    else:
        frozen = agw.frozen
    assert frozen == 0

    if Lpq is None:
        Lpq = agw.ao2mo(mo_coeff)

    if orbs is None:
        orbs = range(agw.nmo)
    else:
        raise NotImplementedError

    nocc = agw.nocc
    nmo = agw.nmo
    nvir = nmo - nocc
    naux = agw.with_df.get_naoaux()

    se_static = agw.build_se_static(
            Lpq=Lpq,
            mo_coeff=mo_coeff,
            mo_energy=mo_energy,
            vhf_df=vhf_df,
    )

    if moments is None:
        th, tp = agw.build_se_moments(
            nmom,
            Lpq=Lpq,
            mo_energy=mo_energy,
            mo_coeff=mo_coeff,
            npoints=npoints,
        )
    else:
        th, tp = moments

    gf, se = agw.solve_dyson(th, tp, se_static)
    conv = True

    # Check moment errors
    error_th = _moment_error(th, se.get_occupied().moment(range(nmom + 1)))
    error_tp = _moment_error(tp, se.get_virtual().moment(range(nmom + 1)))
    logger.debug(
        agw, "Error in moments: occ = %.6g  vir = %.6g", error_th, error_tp
    )

    return conv, gf, se


def _kernel_evgw(
    agw,
    nmom,
    mo_energy,
    mo_coeff,
    moments=None,
    Lpq=None,
    orbs=None,
    vhf_df=False,
    npoints=48,
    max_cycle=50,
    conv_tol=1e-8,
    conv_tol_moms=1e-8,
    diis_space=10,
    g0=False,
    w0=False,
    verbose=logger.NOTE,
):
    """Moment-constrained evGW. Returns the Green's function and self-
    energy as objects using the `GreensFunction` and `SelfEnergy`
    objects from the `agf2` module.

    Parameters
    ----------
    agw : AGW
        AGW object.
    nmom : int
        Maximum moment number to calculate.
    mo_energy : numpy.ndarray
        Molecular orbital energies.
    mo_coeff : numpy.ndarray
        Molecular orbital coefficients.
    Lpq : np.ndarray, optional
        Density-fitted ERI tensor. If None, generate from `agw.ao2mo`.
        Default value is None.
    orbs : list of int, optional
        List of orbitals to include in GW calculation. If None,
        include all orbitals. Default value is None.
    vhf_df : bool, optional
        If True, calculate the static self-energy directly from `Lpq`.
        Default value is False.
    max_cycle : int, optional
        Maximum number of eigenvalue self-consistent cycles. Default
        value is 50.
    conv_tol_gap : float, optional
        Convergence threshold for the maximum change in the HOMO and
        LUMO. Default value is 1e-8.
    conv_tol_moms : float, optional
        Convergence threshold for the norm of the change in the
        moments. Default value is 1e-8.
    diis_space : int, optional
        Number of moments to include in the DIIS extrapolation.
        Default value is 10.
    g0 : bool, optional
        If True, use G0 (i.e. evG0W). Default value is False.
    w0 : bool, optional
        If True, use W0 (i.e. evGW0). Default value is False.

    Returns
    -------
    conv : bool
        Convergence flag. Always True for AGW, returned for
        compatibility with other GW methods.
    gf : pyscf.agf2.GreensFunction
        Green's function object
    se : pyscf.agf2.SelfEnergy
        Self-energy object
    """

    # TODO tests!

    mf = agw._scf
    if agw.frozen is None:
        frozen = 0
    else:
        frozen = agw.frozen
    assert frozen == 0

    if Lpq is None:
        Lpq = agw.ao2mo(mo_coeff)

    if orbs is None:
        orbs = range(agw.nmo)
    else:
        raise NotImplementedError

    if agw.exact_dRPA:
        raise NotImplementedError("exact_dRPA=True only supported for G0W0.")

    if moments is not None:
        raise NotImplementedError("moments keyword argument only supported for G0W0.")

    nocc = agw.nocc
    nmo = agw.nmo
    nvir = nmo - nocc
    naux = agw.with_df.get_naoaux()

    se_static = agw.build_se_static(
            Lpq=Lpq,
            mo_coeff=mo_coeff,
            mo_energy=mo_energy,
            vhf_df=vhf_df,
    )
    mo_energy = mo_energy.copy()
    mo_energy_ref = mo_energy.copy()
    th_prev = tp_prev = np.zeros((nmom + 1, nmo, nmo))

    diis = lib.diis.DIIS()
    diis.space = diis_space

    name = "evG%sW%s" % ("0" if g0 else "", "0" if w0 else "")

    conv = False
    for cycle in range(1, max_cycle + 1):
        logger.info(agw, "%s iteration %d", name, cycle)

        th, tp = rpamoms.build_se_moments(
            agw,
            nmom,
            Lpq,
            mo_energy=(
                mo_energy if not g0 else mo_energy_ref,
                mo_energy if not w0 else mo_energy_ref,
            ),
            npoints=npoints,
        )
        th, tp = diis.update(np.array((th, tp)))

        gf, se = agw.solve_dyson(th, tp, se_static)

        # Check moment errors
        error_th = _moment_error(th, se.get_occupied().moment(range(nmom + 1)))
        error_tp = _moment_error(tp, se.get_virtual().moment(range(nmom + 1)))
        logger.debug(
            agw, "Error in moments: occ = %.6g  vir = %.6g", error_th, error_tp
        )

        # Update the MO energies
        check = set()
        mo_energy_prev = mo_energy.copy()
        for i in range(nmo):
            arg = np.argmax(gf.coupling[i] ** 2)
            mo_energy[i] = gf.energy[arg]
            check.add(arg)
        assert len(check) == nmo

        # Check convergence
        error_homo = abs(mo_energy[nocc - 1] - mo_energy_prev[nocc - 1])
        error_lumo = abs(mo_energy[nocc] - mo_energy_prev[nocc])
        error_th = _moment_error(th, th_prev)
        error_tp = _moment_error(tp, tp_prev)
        th_prev = th.copy()
        tp_prev = tp.copy()
        logger.info(
            agw, "Change in QPs: HOMO = %.6g  LUMO = %.6g", error_homo, error_lumo
        )
        logger.info(
            agw, "Change in Moments: occ = %.6g  vir = %.6g", error_th, error_tp
        )
        if (
            max(error_homo, error_lumo) < conv_tol
            and max(error_th, error_tp) < conv_tol_moms
        ):
            conv = True
            break

    if conv:
        logger.note(agw, "%s converged", name)
    else:
        logger.note(agw, "%s failed to converge", name)

    return conv, gf, se


def _kernel_scgw(
    agw,
    nmom,
    mo_energy,
    mo_coeff,
    moments=None,
    Lpq=None,
    orbs=None,
    vhf_df=False,
    npoints=48,
    max_cycle=50,
    conv_tol=1e-8,
    conv_tol_moms=1e-8,
    diis_space=10,
    g0=False,
    w0=False,
    verbose=logger.NOTE,
):
    """Moment-constrained scGW. Returns the Green's function and self-
    energy as objects using the `GreensFunction` and `SelfEnergy`
    objects from the `agf2` module.

    Parameters
    ----------
    agw : AGW
        AGW object.
    nmom : int
        Maximum moment number to calculate.
    mo_energy : numpy.ndarray
        Molecular orbital energies.
    mo_coeff : numpy.ndarray
        Molecular orbital coefficients.
    Lpq : np.ndarray, optional
        Density-fitted ERI tensor. If None, generate from `agw.ao2mo`.
        Default value is None.
    orbs : list of int, optional
        List of orbitals to include in GW calculation. If None,
        include all orbitals. Default value is None.
    vhf_df : bool, optional
        If True, calculate the static self-energy directly from `Lpq`.
        Default value is False.
    max_cycle : int, optional
        Maximum number of eigenvalue self-consistent cycles. Default
        value is 50.
    conv_tol_gap : float, optional
        Convergence threshold for the maximum change in the HOMO and
        LUMO. Default value is 1e-8.
    conv_tol_moms : float, optional
        Convergence threshold for the norm of the change in the
        moments. Default value is 1e-8.
    diis_space : int, optional
        Number of moments to include in the DIIS extrapolation.
        Default value is 10.
    g0 : bool, optional
        If True, use G0 (i.e. evG0W). Default value is False.
    w0 : bool, optional
        If True, use W0 (i.e. evGW0). Default value is False.

    Returns
    -------
    conv : bool
        Convergence flag. Always True for AGW, returned for
        compatibility with other GW methods.
    gf : pyscf.agf2.GreensFunction
        Green's function object
    se : pyscf.agf2.SelfEnergy
        Self-energy object
    """

    # TODO tests!

    mf = agw._scf
    if agw.frozen is None:
        frozen = 0
    else:
        frozen = agw.frozen
    assert frozen == 0

    if Lpq is not None:
        logger.warn(agw, "Parameter Lpq is not used by self-consistent GW.")
    Lpk = Lia = None

    if orbs is None:
        orbs = range(agw.nmo)
    else:
        raise NotImplementedError

    if agw.exact_dRPA:
        raise NotImplementedError("exact_dRPA=True only supported for G0W0.")

    if moments is not None:
        raise NotImplementedError("moments keyword argument only supported for G0W0.")

    nocc = agw.nocc
    nmo = agw.nmo
    nvir = nmo - nocc
    naux = agw.with_df.get_naoaux()

    se_static = agw.build_se_static(
            Lpq=Lpq,
            mo_coeff=mo_coeff,
            mo_energy=mo_energy,
            vhf_df=vhf_df,
    )
    gf = GreensFunction(mo_energy, np.eye(mo_energy.size))
    gf_ref = gf.copy()
    th_prev = tp_prev = np.zeros((nmom + 1, nmo, nmo))

    diis = lib.diis.DIIS()
    diis.space = diis_space

    name = "G%sW%s" % ("0" if g0 else "", "0" if w0 else "")

    conv = False
    for cycle in range(1, max_cycle + 1):
        logger.info(agw, "%s iteration %d", name, cycle)

        # Rotate ERIs into (MO, QMO)
        if g0:
            mo = np.asarray(mo_coeff, order="F")
            ijslice = (0, nmo, 0, nmo)
            shape = (naux, nmo, nmo)
            out = Lpk if (Lpk is None or Lpk.size >= np.prod(shape)) else None
        else:
            mo = np.asarray(
                np.concatenate([mo_coeff, np.dot(mo_coeff, gf.coupling)], axis=1),
                order="F",
            )
            ijslice = (0, nmo, nmo, nmo + gf.naux)
            shape = (naux, nmo, gf.naux)
            out = Lpk if (Lpk is None or Lpk.size >= np.prod(shape)) else None
        Lpk = _ao2mo.nr_e2(agw.with_df._cderi, mo, ijslice, aosym="s2", out=out)
        Lpk = Lpk.reshape(shape)

        # Rotate ERIs into (QMO occ, QMO vir)
        if w0:
            mo = mo_coeff
            ijslice = (0, nocc, nocc, nmo)
            shape = (naux, nocc, nvir)
            out = Lia if (Lia is None or Lia.size >= np.prod(shape)) else None
        else:
            mo = np.asarray(np.dot(mo_coeff, gf.coupling), order="F")
            nocc_aux = gf.get_occupied().naux
            nvir_aux = gf.get_virtual().naux
            ijslice = (0, nocc_aux, nocc_aux, gf.naux)
            shape = (naux, nocc_aux, nvir_aux)
            out = Lia if (Lia is None or Lia.size >= np.prod(shape)) else None
        Lia = _ao2mo.nr_e2(agw.with_df._cderi, mo, ijslice, aosym="s2", out=out)
        Lia = Lia.reshape(shape)

        th, tp = rpamoms.build_se_moments(
            agw,
            nmom,
            Lpk,
            Lia=Lia,
            mo_energy=(
                gf.energy if not g0 else gf_ref.energy,
                gf.energy if not w0 else gf_ref.energy,
            ),
            mo_occ=(
                _gf_to_occ(gf if not g0 else gf_ref),
                _gf_to_occ(gf if not w0 else gf_ref),
            ),
            npoints=npoints,
        )
        th, tp = diis.update(np.array((th, tp)))

        gf_prev = gf.copy()
        gf, se = agw.solve_dyson(th, tp, se_static)

        # Check moment errors
        error_th = _moment_error(th, se.get_occupied().moment(range(nmom + 1)))
        error_tp = _moment_error(tp, se.get_virtual().moment(range(nmom + 1)))
        logger.debug(
            agw, "Error in moments: occ = %.6g  vir = %.6g", error_th, error_tp
        )

        # Check convergence
        error_homo = abs(
            gf.energy[np.argmax(gf.coupling[nocc - 1] ** 2)]
            - gf_prev.energy[np.argmax(gf_prev.coupling[nocc - 1] ** 2)]
        )
        error_lumo = abs(
            gf.energy[np.argmax(gf.coupling[nocc] ** 2)]
            - gf_prev.energy[np.argmax(gf_prev.coupling[nocc] ** 2)]
        )
        error_th = _moment_error(th, th_prev)
        error_tp = _moment_error(tp, tp_prev)
        th_prev = th.copy()
        tp_prev = tp.copy()
        logger.info(
            agw, "Change in QPs: HOMO = %.6g  LUMO = %.6g", error_homo, error_lumo
        )
        logger.info(
            agw, "Change in Moments: occ = %.6g  vir = %.6g", error_th, error_tp
        )
        if (
            max(error_homo, error_lumo) < conv_tol
            and max(error_th, error_tp) < conv_tol_moms
        ):
            conv = True
            break

    if conv:
        logger.note(agw, "%s converged", name)
    else:
        logger.note(agw, "%s failed to converge", name)

    return conv, gf, se


def _moment_error(t, t_prev):
    """Compute the scaled error in the moments."""

    error = 0
    for a, b in zip(t, t_prev):
        a = a / max(np.max(np.abs(a)), 1)
        b = b / max(np.max(np.abs(b)), 1)
        error = max(error, np.max(np.abs(a - b)))

    return error


def _gf_to_occ(gf):
    """Convert a GF to an MO occupancy."""

    gf_occ = gf.get_occupied()

    occ = np.zeros((gf.naux,))
    occ[: gf_occ.naux] = np.sum(gf_occ.coupling**2, axis=0) * 2.0

    return occ


def build_se_static(agw, Lpq=None, vhf_df=False, mo_coeff=None, mo_energy=None):
    """Build the static part of the self-energy, including the Fock
    matrix.

    Parameters
    ----------
    agw : AGW
        AGW object.
    Lpq : np.ndarray, optional
        Density-fitted ERI tensor. If None, generate from `agw.ao2mo`.
        Default value is None.
    vhf_df : bool, optional
        If True, calculate the static self-energy directly from `Lpq`.
        Default value is False.
    mo_coeff : numpy.ndarray, optional
        Molecular orbital coefficients. If None, use array from
        `agw._scf`. Default value is None.
    mo_energy : numpy.ndarray, optional
        Molecular orbital energies. If None, use array from
        `agw._scf`. Default value is None.

    Returns
    -------
    se_static : np.ndarray
        Static part of the self-energy. If `agw.diag_sigma`,
        non-diagonal elements are set to zero.
    """

    if mo_coeff is None:
        mo_coeff = agw._scf.mo_coeff
    if mo_energy is None:
        mo_energy = agw._scf.mo_energy
    if Lpq is None and vhf_df:
        Lpq = agw.ao2mo(mo_coeff)

    v_mf = agw._scf.get_veff() - agw._scf.get_j()
    v_mf = einsum("pq,pi,qj->ij", v_mf, mo_coeff, mo_coeff)

    # v_hf from DFT/HF density
    if vhf_df:
        sc = np.dot(agw._scf.get_ovlp(), mo_coeff)
        dm = einsum("pq,pi,qj->ij", agw._scf.make_rdm1(mo_coeff=mo_coeff), sc, sc)
        tmp = einsum("Qik,kl->Qil", Lpq, dm)
        vk = -einsum("Qil,Qlj->ij", tmp, Lpq) * 0.5
    else:
        dm = agw._scf.make_rdm1(mo_coeff=mo_coeff)
        vk = scf.hf.SCF.get_veff(agw._scf, agw.mol, dm) - scf.hf.SCF.get_j(
            agw._scf, agw.mol, dm
        )
        vk = einsum("pq,pi,qj->ij", vk, mo_coeff, mo_coeff)

    se_static = vk - v_mf

    if agw.diag_sigma:
        se_static = np.diag(np.diag(se_static))

    se_static += np.diag(mo_energy)

    return se_static


def build_se_moments(
    agw,
    nmom,
    Lpq=None,
    mo_energy=None,
    mo_coeff=None,
    npoints=48,
    debug=DEBUG,
):
    """Build the density-density response moments.

    Parameters
    ----------
    agw : AGW
        AGW object.
    nmom : int
        Maximum moment number to calculate.
    Lpq : np.ndarray, optional
        Density-fitted ERI tensor. If None, generate from `agw.ao2mo`.
        Default value is None.
    mo_energy : numpy.ndarray or tuple of numpy.ndarray, optional
        Molecular orbital energies. If None, use array from `agw._scf`.
        If tuple, then first element corresponds to the MO energies
        for the Green's function, and the second element for the
        screened Coulomb interaction. Default value is None.
    mo_coeff : numpy.ndarray, optional
        Molecular orbital coefficients. If None, use array from
        `agw._scf`. Default value is None.
    npoints : int, optional
        Number of quadrature points to use. Default value is 48.

    Returns
    -------
    hole_moms : np.ndarray
        Moments of the hole self-energy. If `agw.diag_sigma`,
        non-diagonal elements are set to zero.
    part_moms : np.ndarray
        Moments of the particle self-energy. If `agw.diag_sigma`,
        non-diagonal elements are set to zero.
    """

    if not agw.exact_dRPA and Lpq is not None:
        # Optimised routine - improve later
        return rpamoms.build_se_moments(
            agw,
            nmom,
            Lpq,
            mo_energy=mo_energy,
            npoints=npoints,
        )

    if mo_energy is None:
        mo_energy_g = mo_energy_w = agw._scf.mo_energy
    elif isinstance(mo_energy, tuple):
        mo_energy_g, mo_energy_w = mo_energy
    else:
        mo_energy_g = mo_energy_w = mo_energy
    if Lpq is None:
        Lpq = agw.ao2mo(mo_coeff)

    nmo = agw.nmo
    nocc = agw.nocc

    logger.debug(agw, "Building moments up to nmom = %d", nmom)
    logger.debug(
        agw, "Computing the moments of the tild_eta (~ screened coulomb moments)"
    )
    tild_etas = rpamoms.get_tilde_dd_moms(
        agw._scf,
        nmom,
        Lpq=Lpq,
        use_ri=not agw.exact_dRPA,
        npoints=npoints,
    )

    logger.debug(agw, "Contracting dd moments with second coulomb interaction")
    if agw.diag_sigma:
        # Simplifications due to only constructing the diagonal self-energy
        tild_sigma = np.zeros((nmo, nmom + 1, nmo))
        for x in range(nmo):
            Lpx = Lpq[:, :, x]
            tild_sigma[x] = einsum("Pp,Qp,nQP->np", Lpx, Lpx, tild_etas)

        logger.debug(agw, "Forming particle and hole self-energy")
        part_moms = []
        hole_moms = []
        moms = np.arange(nmom + 1)
        for n in range(nmom + 1):
            fp = scipy.special.binom(n, moms)
            fh = fp * (-1) ** moms
            e = np.power.outer(mo_energy_g, n - moms)
            th = einsum("t,kt,ktp->p", fh, e[:nocc], tild_sigma[:nocc])
            tp = einsum("t,ct,ctp->p", fp, e[nocc:], tild_sigma[nocc:])
            hole_moms.append(np.diag(th))
            part_moms.append(np.diag(tp))

    else:
        tild_sigma = np.zeros((nmo, nmom + 1, nmo, nmo))
        for x in range(nmo):
            Lpx = Lpq[:, :, x]
            tild_sigma[x] = einsum("Pq,Qp,nQP->npq", Lpx, Lpx, tild_etas)

        logger.debug(agw, "Forming particle and hole self-energy")
        part_moms = []
        hole_moms = []
        moms = np.arange(nmom + 1)
        for n in range(nmom + 1):
            fp = scipy.special.binom(n, moms)
            fh = fp * (-1) ** moms
            e = np.power.outer(mo_energy_g, n - moms)
            th = einsum("t,kt,ktpq->pq", fh, e[:nocc], tild_sigma[:nocc])
            tp = einsum("t,ct,ctpq->pq", fp, e[nocc:], tild_sigma[nocc:])
            hole_moms.append(th)
            part_moms.append(tp)

    # if agw.diag_sigma:
    #    hole_moms = [np.diag(np.diag(t)) for t in hole_moms]
    #    part_moms = [np.diag(np.diag(t)) for t in part_moms]

    hole_moms = [0.5 * (t + t.T) for t in hole_moms]
    part_moms = [0.5 * (t + t.T) for t in part_moms]

    if debug:
        # Log the definiteness of the moments
        mmin = lambda x: ("%12.6g" % min(x)) if len(x) else ""
        mmax = lambda x: ("%12.6g" % max(x)) if len(x) else ""
        logger.debug(
            agw,
            "%12s %12s %12s %12s %12s",
            "Moment",
            "min neg",
            "max neg",
            "min pos",
            "max pos",
        )
        for n in range(nmom + 1):
            w = np.linalg.eigvalsh(hole_moms[n])
            vals = (mmin(w[w < 0]), mmax(w[w < 0]), mmin(w[w >= 0]), mmax(w[w >= 0]))
            logger.debug(agw, "hole %-7d %12s %12s %12s %12s", n, *vals)
        for n in range(nmom + 1):
            w = np.linalg.eigvalsh(part_moms[n])
            vals = (mmin(w[w < 0]), mmax(w[w < 0]), mmin(w[w >= 0]), mmax(w[w >= 0]))
            logger.debug(agw, "part %-7d %12s %12s %12s %12s", n, *vals)

    return hole_moms, part_moms


def solve_dyson(agw, hole_moms, part_moms, se_static):
    """Solve the Dyson equation due to a self-energy resulting from
    a list of hole and particle moments, along with a static
    contribution to the self-energy.

    Also finds a chemical potential which best staisfies the physical
    number of electrons. If `agw.optimise_chempot`, this will shift
    the self-energy poles relative to the Green's function, which may
    be considered a partial self-consistency. Otherwise, just find a
    chemical potential according to Aufbau principal and do not allow
    relative difference in chemical potentials in SE and GF.

    Parameters
    ----------
    agw : AGW
        AGW object.
    hole_moms : numpy.ndarray
        Moments of the hole self-energy.
    part_moms : numpy.ndarray
        Moments of the particle self-energy.
    se_static : numpy.ndaray
        Static part of the self-energy, including the Fock matrix.

    Returns
    -------
    gf : agf2.GreensFunction
        Green's function object
    se : agf2.SelfEnergy
        Self-energy object
    """

    se_occ = block_lanczos_se(se_static, hole_moms)
    se_vir = block_lanczos_se(se_static, part_moms)
    se = combine(se_occ, se_vir)

    if agw.optimise_chempot:
        # Shift the self-energy poles w.r.t the Green's function
        se, opt = chempot.minimize_chempot(se, se_static, agw.mol.nelectron)

    gf = se.get_greens_function(se_static)

    try:
        cpt, error = chempot.binsearch_chempot(
            (gf.energy, gf.coupling), gf.nphys, agw.mol.nelectron
        )
    except:
        cpt = 0.5 * (
            mo_energy[agw._scf.mo_occ > 0].max() + mo_energy[agw._scf.mo_occ == 0].min()
        )
        gf.chempot = cpt
        error = np.trace(gf.make_rdm1()) - agw.mol.nelectron

    se.chempot = cpt
    gf.chempot = cpt
    logger.info(agw, "Error in number of electrons: %.5g", error)

    return gf, se


def block_lanczos_se(se_static, se_moms):
    """Transform a set of moments of the self-energy to a pole
    representation.

    Parameters
    ----------
    se_static : np.ndarray (n, n)
        Static part of the self-energy
    se_moms : np.ndarray (k, n, n)
        Moments of the self-energy, for j iterations of block Lanczos
        the first k=2*j+2 moments are required.

    Returns
    -------
    se : agf2.SelfEnergy
        Self-energy object
    """

    try:
        from dyson import MBLSE, NullLogger
    except:
        try:
            from dyson import BlockLanczosSymmSE
        except:
            raise ValueError(
                "https://github.com/obackhouse/dyson-compression "
                "is deprecated in favour of "
                "https://github.com/BoothGroup/dyson"
            )
        raise ValueError("Missing dependency: " "https://github.com/BoothGroup/dyson")

    solver = MBLSE(se_static, np.array(se_moms), log=NullLogger())
    solver.kernel()
    e_aux, v_aux = solver.get_auxiliaries()

    se = SelfEnergy(e_aux, v_aux)

    return se


class AGW(lib.StreamObject):
    """Moment-conserved GW.

    Parameters
    ----------
    mf : pyscf.scf.SCF
        PySCF mean-field object.
    frozen : int or tuple of int, optional
        Frozen orbitals. Default value is None.

    Attributes
    ----------
    diag_sigma : bool
        If True, assume a diagonal approximation in the self-energy.
        Default value is False.
    exact_dRPA : bool
        If True, exactly solve the frequency integral for the DD
        moments. Exact solution scales as N^6 in system sizes, while
        the numerical integration scales as N^4. Default value is
        False.
    optimise_chempot : bool
        If True, optimise the number of electrons by applying a
        constant shift in the poles of the self-energy, thereby
        applying a different relative chemical potential in the self-
        energy and Green's functions. Default value is False.
    gf : pyscf.agf2.GreensFunction
        Green's function object
    sigma : pyscf.agf2.SelfEnergy
        Self-energy object
    """

    diag_sigma = getattr(__config__, "gw_gw_GW_diag_sigma", False)
    exact_dRPA = getattr(__config__, "gw_gw_GW_exact_dRPA", False)
    optimise_chempot = getattr(__config__, "gw_gw_GW_optimise_chempot", False)

    def __init__(self, mf, frozen=None):
        self.mol = mf.mol
        self._scf = mf
        self.verbose = self.mol.verbose
        self.stdout = self.mol.stdout
        self.max_memory = mf.max_memory

        self.frozen = frozen
        if not (self.frozen is None or self.frozen == 0):
            raise NotImplementedError

        # moment-GW must use density fitting integrals for the N^4 algorithm
        if getattr(mf, "with_df", None):
            self.with_df = mf.with_df
        else:
            self.with_df = df.DF(mf.mol)
            self.with_df.auxbasis = df.make_auxbasis(mf.mol, mp2fit=True)

        ##################################################
        # don't modify the following attributes, they are not input options
        self.mo_coeff = mf.mo_coeff
        self.mo_occ = mf.mo_occ
        self._nocc = None
        self._nmo = None
        self.sigma = None
        self.gf = None

        keys = set(("diag_sigma", "exact_dRPA", "optimise_chempot"))
        self._keys = set(self.__dict__.keys()).union(keys)

    def dump_flags(self):
        log = logger.Logger(self.stdout, self.verbose)
        log.info("")
        log.info("******** %s ********", self.__class__)
        log.info("method = %s", self.__class__.__name__)
        nocc = self.nocc
        nvir = self.nmo - nocc
        log.info("Moment-constrained GW nocc = %d, nvir = %d", nocc, nvir)
        if self.frozen is not None:
            log.info("frozen = %s", self.frozen)
        log.info(
            "Use N^6 exact computation of self-energy moments = %s", self.exact_dRPA
        )
        log.info("Use diagonal self-energy in QP eqn = %s", self.diag_sigma)
        return self

    @property
    def mo_energy(self):
        # TODO should we prune low-weighted energies?
        return self.gf.mo_energy

    @property
    def nocc(self):
        return self.get_nocc()

    @nocc.setter
    def nocc(self, n):
        self._nocc = n

    @property
    def nmo(self):
        return self.get_nmo()

    @nmo.setter
    def nmo(self, n):
        self._nmo = n

    get_nocc = get_nocc
    get_nmo = get_nmo
    get_frozen_mask = get_frozen_mask

    build_se_static = build_se_static
    build_se_moments = build_se_moments
    solve_dyson = solve_dyson

    def kernel(
        self,
        nmom=1,
        moments=None,
        mo_energy=None,
        mo_coeff=None,
        Lpq=None,
        orbs=None,
        vhf_df=False,
        roots=10,
        npoints=48,
        method="g0w0",
        **kwargs
    ):
        if mo_coeff is None:
            mo_coeff = self._scf.mo_coeff
        if mo_energy is None:
            mo_energy = self._scf.mo_energy

        cput0 = (logger.process_clock(), logger.perf_counter())
        self.dump_flags()
        logger.info(
            self, "Number of moments to compute in self-energy expansion = %s", nmom
        )

        if method.lower() == "g0w0":
            kernel = _kernel
        elif method.lower() in ("scgw", "gw"):
            kernel = _kernel_scgw
        elif method.lower() in ("scgw0", "gw0"):
            kernel = partial(_kernel_scgw, w0=True)
        elif method.lower() in ("scg0w", "g0w"):
            kernel = partial(_kernel_scgw, g0=True)
        elif method.lower() == "evgw":
            kernel = _kernel_evgw
        elif method.lower() == "evgw0":
            kernel = partial(_kernel_evgw, w0=True)
        elif method.lower() == "evg0w":
            kernel = partial(_kernel_evgw, g0=True)
        elif method.lower() == "evg0w0":
            kernel = partial(_kernel_evgw, g0=True, w0=True)
        else:
            raise ValueError(method)

        self.converged, self.gf, self.sigma = kernel(
            self,
            nmom,
            mo_energy,
            mo_coeff,
            Lpq=Lpq,
            moments=moments,
            orbs=orbs,
            vhf_df=vhf_df,
            npoints=npoints,
            verbose=self.verbose,
            **kwargs
        )

        gf_occ = self.gf.get_occupied()
        gf_occ.remove_uncoupled(tol=1e-1)
        for n in range(min(roots, gf_occ.naux)):
            en = gf_occ.energy[-(n + 1)]
            vn = gf_occ.coupling[:, -(n + 1)]
            qpwt = np.linalg.norm(vn) ** 2
            logger.note(
                self, "IP energy level %d E = %.16g  QP weight = %0.6g", n, en, qpwt
            )

        gf_vir = self.gf.get_virtual()
        gf_vir.remove_uncoupled(tol=1e-1)
        for n in range(min(roots, gf_vir.naux)):
            en = gf_vir.energy[n]
            vn = gf_vir.coupling[:, n]
            qpwt = np.linalg.norm(vn) ** 2
            logger.note(
                self, "EA energy level %d E = %.16g  QP weight = %0.6g", n, en, qpwt
            )

        logger.timer(self, "Moment GW", *cput0)

        return self.converged, self.gf, self.sigma

    def ao2mo(self, mo_coeff=None):
        """Get MO basis density-fitted integrals."""

        if mo_coeff is None:
            mo_coeff = self.mo_coeff
        nmo = self.nmo
        naux = self.with_df.get_naoaux()
        mem_incore = (2 * nmo**2 * naux) * 8 / 1e6
        mem_now = lib.current_memory()[0]

        mo = numpy.asarray(mo_coeff, order="F")
        ijslice = (0, nmo, 0, nmo)
        Lpq = None
        if (mem_incore + mem_now < 0.99 * self.max_memory) or self.mol.incore_anyway:
            Lpq = _ao2mo.nr_e2(self.with_df._cderi, mo, ijslice, aosym="s2", out=Lpq)
            return Lpq.reshape(naux, nmo, nmo)
        else:
            logger.warn(self, "Memory may not be enough!")
            raise NotImplementedError


del DEBUG


if __name__ == "__main__":
    from pyscf import gto, dft

    mol = gto.Mole()
    mol.verbose = 4
    mol.atom = [
        [8, (0.0, 0.0, 0.0)],
        [1, (0.0, -0.7571, 0.5861)],
        [1, (0.0, 0.7571, 0.5861)],
    ]
    mol.basis = "def2-svp"
    mol.build()

    mf = dft.RKS(mol)
    mf = mf.density_fit()
    mf.xc = "pbe"
    mf.kernel()

    nocc = mol.nelectron // 2
    nmo = mf.mo_energy.size
    nvir = nmo - nocc

    gw = AGW(mf)
    gw.exact_dRPA = True
    gw.kernel(nmom=5)
