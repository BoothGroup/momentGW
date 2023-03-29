"""
Construct RPA moments.
"""

import numpy as np
import scipy.special
from pyscf import lib
from pyscf.agf2 import mpi_helper
from vayesta.core.vlog import NoLogger
from vayesta.rpa import ssRIRPA, ssRPA
from vayesta.rpa.rirpa import momzero_NI

# TODO silence Vayesta


def compress_low_rank(ri_l, ri_r, tol=1e-12):
    """Perform the low-rank compression."""

    naux_init = ri_l.shape[0]

    u, s, v = np.linalg.svd(ri_l, full_matrices=False)
    nwant = sum(s > tol)
    rot = u[:, :nwant]
    ri_l = np.dot(rot.T, ri_l)
    ri_r = np.dot(rot.T, ri_r)

    u, s, v = np.linalg.svd(ri_r, full_matrices=False)
    nwant = sum(s > tol)
    rot = u[:, :nwant]
    ri_l = np.dot(rot.T, ri_l)
    ri_r = np.dot(rot.T, ri_r)

    return ri_l, ri_r


def build_se_moments_drpa_exact(
    gw,
    nmom_max,
    Lpq,
    mo_energy=None,
    ainit=10,
):
    """
    Compute the self-energy moments using exact dRPA.  Scales as
    the sixth power of the number of orbitals.

    Parameters
    ----------
    gw : BaseGW
        GW object.
    nmom_max : int
        Maximum moment number to calculate.
    Lpq : numpy.ndarra
        Density-fitted ERI tensor.
    exact : bool, optional
        Use exact dRPA at O(N^6) cost.  Default value is `False`.
    mo_energy : numpy.ndarray, optional
        Molecular orbital energies.  Default value is that of
        `gw._scf.mo_energy`.
    aint : int, optional
        Initial `a` value, see `Vayesta` for more details.  Default
        value is 10.

    Returns
    -------
    se_moments_hole : numpy.ndarray
        Moments of the hole self-energy. If `self.diagonal_se`,
        non-diagonal elements are set to zero.
    se_moments_part : numpy.ndarray
        Moments of the particle self-energy. If `self.diagonal_se`,
        non-diagonal elements are set to zero.
    """

    if mo_energy is None:
        mo_energy = gw._scf.mo_energy

    nmo = gw.nmo
    nocc = gw.nocc
    nov = nocc * (nmo - nocc)

    # Get 3c integrals
    Lia = Lpq[:, :nocc, nocc:]
    rot = np.concatenate([Lia.reshape(-1, nov)] * 2, axis=1)

    hole_moms = np.zeros((nmom_max + 1, nmo, nmo))
    part_moms = np.zeros((nmom_max + 1, nmo, nmo))

    # Get DD moments
    rpa = ssRPA(mf)
    erpa = rpa.kernel()
    tild_etas = rpa.gen_tild_etas(nmom_max)
    tild_etas = lib.einsum("nij,pi,qj->npq", tild_etas, rot, rot)

    # Construct the SE moments
    if gw.diagonal_se:
        tild_sigma = np.zeros((nmo, nmom_max + 1, nmo))
        for x in range(nmo):
            Lpx = Lpq[p0:p1, :, x]
            Lqx = Lpq[q0:q1, :, x]
            tild_sigma[x] = lib.einsum("Pp,Qp,nPQ->np", Lpx, Lqx, tild_etas)
        moms = np.arange(nmom_max + 1)
        for n in range(nmom_max + 1):
            fp = scipy.special.binom(n, moms)
            fh = fp * (-1) ** moms
            eon = np.power.outer(mo_energy[:nocc], n - moms)
            evn = np.power.outer(mo_energy[nocc:], n - moms)
            th = lib.einsum("t,kt,ktp->p", fh, eon, tild_sigma[:nocc])
            tp = lib.einsum("t,ct,ctp->p", fp, evn, tild_sigma[nocc:])
            hole_moms[n] += np.diag(th)
            part_moms[n] += np.diag(tp)
    else:
        tild_sigma = np.zeros((nmo, nmom_max + 1, nmo, nmo))
        moms = np.arange(nmom_max + 1)
        for x in range(nmo):
            Lpx = Lpq[:, :, x]
            Lqx = Lpq[:, :, x]
            tild_sigma[x] = lib.einsum("Pp,Qq,nPQ->npq", Lpx, Lqx, tild_etas)
        for n in range(nmom_max + 1):
            fp = scipy.special.binom(n, moms)
            fh = fp * (-1) ** moms
            eon = np.power.outer(mo_energy[:nocc], n - moms)
            evn = np.power.outer(mo_energy[nocc:], n - moms)
            th = lib.einsum("t,kt,ktpq->pq", fh, eon, tild_sigma[:nocc])
            tp = lib.einsum("t,ct,ctpq->pq", fp, evn, tild_sigma[nocc:])
            hole_moms[n] += th
            part_moms[n] += tp

    hole_moms = 0.5 * (hole_moms + hole_moms.swapaxes(1, 2))
    part_moms = 0.5 * (part_moms + part_moms.swapaxes(1, 2))

    return hole_moms, part_moms


def build_se_moments_drpa(
    gw,
    nmom_max,
    Lpq,
    Lia=None,
    mo_energy=None,
    mo_occ=None,
    ainit=10,
    compress=0,
):
    """
    Compute the self-energy moments using dRPA and numerical
    integration.

    Parameters
    ----------
    gw : BaseGW
        GW object.
    nmom_max : int
        Maximum moment number to calculate.
    Lpq : numpy.ndarray
        Density-fitted ERI tensor. `p` is in the basis of MOs, `q` is
        in the basis of the Green's function.
    Lia : numpy.ndarray, optional
        Density-fitted ERI tensor for the occupied-virtual slice. `i`
        and `a` are in the basis of the screened Coulomb interaction.
    mo_energy : numpy.ndarray or tuple of numpy.ndarray, optional
        Molecular orbital energies.  If a tuple is passed, the first
        element corresponds to the Green's function basis and the
        second to the screened Coulomb interaction.  Default value is
        that of `gw._scf.mo_energy`.
    mo_occ : numpy.ndarray or tuple of numpy.ndarray, optional
        Molecular orbital occupancies.  If a tuple is passed, the first
        element corresponds to the Green's function basis and the
        second to the screened Coulomb interaction.  Default value is
        that of `gw._scf.mo_occ`.
    aint : int, optional
        Initial `a` value, see `Vayesta` for more details.  Default
        value is 10.
    compress : int, optional
        How thoroughly to attempt compression of the low-rank
        representations of various matrices.
        Thresholds are:
        - above 0: Compress representation of (A+B)(A-B) once
          constructed, prior to main calculation.
        - above 3: Compress representations of A+B and A-B separately
          prior to constructing (A+B)(A-B) or (A+B)^{-1}
        - above 5: Compress representation of (A+B)^{-1} prior to
          contracting. This is basically never worthwhile.
        Note that in all cases these compressions will have
        computational cost O(N_{aux}^2 ov), the same as our later
        computations, and so a tradeoff must be made between reducing
        the N_{aux} in later calculations vs the cost of compression.
        Default value is 0.

    Returns
    -------
    se_moments_hole : numpy.ndarray
        Moments of the hole self-energy. If `self.diagonal_se`,
        non-diagonal elements are set to zero.
    se_moments_part : numpy.ndarray
        Moments of the particle self-energy. If `self.diagonal_se`,
        non-diagonal elements are set to zero.
    """

    if mo_energy is None:
        mo_energy_g = mo_energy_w = gw._scf.mo_energy
    elif isinstance(mo_energy, tuple):
        mo_energy_g, mo_energy_w = mo_energy
    else:
        mo_energy_g = mo_energy_w = mo_energy

    if mo_occ is None:
        mo_occ_g = mo_occ_w = gw._scf.mo_occ
    elif isinstance(mo_occ, tuple):
        mo_occ_g, mo_occ_w = mo_occ
    else:
        mo_occ_g = mo_occ_w = mo_occ

    vlog = NoLogger()

    nmo = gw.nmo
    naux = gw.with_df.get_naoaux()

    # Get 3c integrals
    if Lia is None:
        Lia = Lpq[:, mo_occ_w > 0][:, :, mo_occ_w == 0]

    eo = mo_energy_w[mo_occ_w > 0]
    ev = mo_energy_w[mo_occ_w == 0]
    nov = eo.size * ev.size

    # Get rotation matrix and A+B
    apb = Lia.reshape(naux, nov) * np.sqrt(2)
    apb = np.concatenate([apb, apb], axis=1)
    if compress > 3:
        apb = compress_low_rank(apb, apb)

    # Get compressed MP
    d = lib.direct_sum("a-i->ia", ev, eo).ravel()
    d = np.concatenate([d, d])
    mp_l = apb[0] * d[None]
    mp_r = apb[1]
    if compress > 0:
        mp_l, mp_r = compress_low_rank(mp_l, mp_r)

    # Construct inverse
    u = np.dot(apb[1] / d[None], apb[0].T)
    u = np.linalg.inv(np.eye(u.shape[0]) + u)
    u, s, v = np.linalg.svd(u)
    urt_l = u * s[None] ** 0.5
    urt_r = v * s[:, None] ** 0.5
    apb_inv_l = np.dot(urt_l.T, apb[0]) / d[None]
    apb_inv_r = np.dot(urt_r, apb[1]) / d[None]
    if compress > 5:
        apb_inv_l, apb_inv_r = compress_low_rank(apb_inv_l, apb_inv_r)

    hole_moms = np.zeros((nmom_max + 1, nmo, nmo))
    part_moms = np.zeros((nmom_max + 1, nmo, nmo))
    for p0, p1 in mpi_helper.prange(0, naux, naux):
        # Get rotation matrix
        rot = Lia[p0:p1].reshape(p1 - p0, nov)
        rot = np.concatenate([rot, rot], axis=1)

        # Perform the offset integral
        offset = momzero_NI.MomzeroOffsetCalcGaussLag(d, mp_l, mp_r, rot, gw.npoints, vlog)
        estval, offset_err = offset.kernel()
        integral_offset = rot * d[None] + estval

        # Perform the rest of the integral
        worker = momzero_NI.MomzeroDeductHigherOrder(d, mp_l, mp_r, rot, gw.npoints, vlog)
        a = worker.opt_quadrature_diag(ainit)
        quad = worker.get_quad(a)
        integral = np.zeros((p1 - p0, nov * 2))
        integral_h = np.zeros((p1 - p0, nov * 2))
        integral_q = np.zeros((p1 - p0, nov * 2))
        for i, (point, weight) in enumerate(zip(*quad)):
            f = 1.0 / (d**2 + point**2)
            q = np.dot(mp_r * f[None], mp_l.T)  # NOTE CPU bottleneck
            lrot = rot * f[None]
            val_aux = np.linalg.inv(np.eye(q.shape[0]) + q) - np.eye(q.shape[0])
            contrib = np.linalg.multi_dot((lrot, mp_l.T, val_aux, mp_r))
            contrib *= f[None]
            contrib *= point**2
            contrib /= np.pi
            integral += weight * contrib
            if i % 2 == 0:
                integral_h += 2 * weight * contrib
            if i % 4 == 0:
                integral_q += 4 * weight * contrib
        a = np.linalg.norm(integral_q - integral)
        b = np.linalg.norm(integral_h - integral)
        err = worker.calculate_error(a, b)

        # Get the zeroth order moment
        integral_part = integral + integral_offset
        t0 = integral_part / d[None]
        t0 -= np.linalg.multi_dot((integral_part, apb_inv_l.T, apb_inv_r))

        # Get the errors
        pinv_norm = np.sum(d**-2)
        pinv_norm += 2.0 * apb_inv_l * apb_inv_r / d[None]
        pinv_norm += np.linalg.norm((apb_inv_l, apb_inv_r)) ** 4
        pinv_norm **= 0.5
        t0_err = err * pinv_norm
        # self.check_errors(t0_err, rot.size)
        # self.test_eta0_error(t0, rot, apb, amb)

        # Get the first order moment
        t1 = rot * d[None]

        # Populate the moments
        moments = np.zeros((nmom_max + 1, *t0.shape))
        moments[0] = t0
        moments[1] = t1
        for i in range(2, nmom_max + 1):
            moments[i] = moments[i - 2] * d[None] ** 2
            moments[i] += np.linalg.multi_dot(
                (moments[i - 2], mp_r.T, mp_l)
            )  # NOTE needs the full mp

        for q0, q1 in lib.prange(0, naux, 500):
            # Rotate right side
            rotq = Lia[q0:q1].reshape(q1 - q0, nov)
            rotq = np.concatenate([rotq, rotq], axis=1)
            tild_etas = lib.einsum("nPk,Qk->nPQ", moments, rotq)  # NOTE likely RAM bottleneck

            # Construct the SE moments
            if gw.diagonal_se:
                tild_sigma = np.zeros((mo_energy_g.size, nmom_max + 1, nmo))
                for x in range(mo_energy_g.size):
                    Lpx = Lpq[p0:p1, :, x]
                    Lqx = Lpq[q0:q1, :, x]
                    tild_sigma[x] = lib.einsum("Pp,Qp,nPQ->np", Lpx, Lqx, tild_etas)
                moms = np.arange(nmom_max + 1)
                for n in range(nmom_max + 1):
                    fp = scipy.special.binom(n, moms)
                    fh = fp * (-1) ** moms
                    eon = np.power.outer(mo_energy_g[mo_occ_g > 0], n - moms)
                    evn = np.power.outer(mo_energy_g[mo_occ_g == 0], n - moms)
                    th = lib.einsum("t,kt,ktp->p", fh, eon, tild_sigma[mo_occ_g > 0])
                    tp = lib.einsum("t,ct,ctp->p", fp, evn, tild_sigma[mo_occ_g == 0])
                    hole_moms[n] += np.diag(th)
                    part_moms[n] += np.diag(tp)
            else:
                tild_sigma = np.zeros((mo_energy_g.size, nmom_max + 1, nmo, nmo))
                moms = np.arange(nmom_max + 1)
                for x in range(mo_energy_g.size):
                    Lpx = Lpq[p0:p1, :, x]
                    Lqx = Lpq[q0:q1, :, x]
                    tild_sigma[x] = lib.einsum("Pp,Qq,nPQ->npq", Lpx, Lqx, tild_etas)
                for n in range(nmom_max + 1):
                    fp = scipy.special.binom(n, moms)
                    fh = fp * (-1) ** moms
                    eon = np.power.outer(mo_energy_g[mo_occ_g > 0], n - moms)
                    evn = np.power.outer(mo_energy_g[mo_occ_g == 0], n - moms)
                    th = lib.einsum("t,kt,ktpq->pq", fh, eon, tild_sigma[mo_occ_g > 0])
                    tp = lib.einsum("t,ct,ctpq->pq", fp, evn, tild_sigma[mo_occ_g == 0])
                    hole_moms[n] += th
                    part_moms[n] += tp

    mpi_helper.barrier()
    hole_moms = mpi_helper.allreduce(hole_moms)
    part_moms = mpi_helper.allreduce(part_moms)

    hole_moms = 0.5 * (hole_moms + hole_moms.swapaxes(1, 2))
    part_moms = 0.5 * (part_moms + part_moms.swapaxes(1, 2))

    return hole_moms, part_moms
