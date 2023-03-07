import vayesta.rpa
import numpy as np
import scipy.special
from pyscf import lib
from pyscf.lib import logger
from vayesta.core.vlog import NoLogger
from vayesta.core.util import dot
from vayesta.rpa.rirpa import momzero_NI
from pyscf.agf2 import mpi_helper

def get_tilde_dd_moms(mf, max_moment, Lpq=None, use_ri=True, npoints=48):
    """Given system specification, return the rpa dd moments contracted with cderi for the density fitted coulomb
    integrals."""
    rot = get_cderi_ph_rot(mf)

    if use_ri:
        moms = np.array(get_dd_moments_rirpa(mf, max_moment, rot, npoints=npoints, Lpq=Lpq))
    else:
        moms = np.array(get_dd_moments_rpa(mf, max_moment, rot))
    return moms

def get_cderi_ph_rot(mf):
    # Define the desired rotation of our particle-hole basis; we're assuming we aren't chunking the RI basis.
    # Get in AO basis.
    cderi = lib.unpack_tril(next(mf.with_df.loop(blksize=mf.with_df.get_naoaux())))
    # Transform into particle-hole excitations.
    def get_ph(_cderi, co, cv):
        #einsum("npq,pi,qa->nia")
        temp = np.tensordot(np.tensordot(_cderi, co, [[1], [0]]), cv, [[1],[0]])
        return temp.reshape((temp.shape[0], -1))

    if len(mf.mo_coeff.shape) < 3:
        # RHF
        nocc = int(sum(mf.mo_occ) / 2)
        rot = get_ph(cderi, mf.mo_coeff[:, :nocc], mf.mo_coeff[:, nocc:])
        rot = np.concatenate([rot, rot], axis=1)
    else:
        # UHF
        nocc = int(sum(mf.mo_occ.T))
        rots = [get_ph(cderi, mf.mo_coeff[i, :, :nocc[i]], mf.mo_coeff[i, :, nocc[i]:]) for i in [0,1]]
        rot = np.concatenate(rots, axis=1)
    return rot

def get_dd_moments_rpa(mf, max_moment, rot):
    rpa = vayesta.rpa.ssRPA(mf)
    erpa = rpa.kernel()
    moms = rpa.gen_moms(max_moment)
    moms = [moms[x] for x in range(max_moment+1)]
    # Now need to rotate into desired basis.
    moms = [dot(rot, x, rot.T) for x in moms]
    return moms

def get_dd_moments_rirpa(mf, max_moment, rot, npoints, Lpq=None):
    myrirpa = vayesta.rpa.ssRIRPA(mf, Lpq=Lpq)

    moms = myrirpa.kernel_moms(max_moment, rot, npoints=npoints)[0]
    # Just need to project the RHS and we're done.
    return [dot(x, rot.T) for x in moms]

def compress_low_rank(ri_l, ri_r, tol=1e-12, log=None, name=None):
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

#def _log_memory(agw, message, shape):
#    if mpi_helper.size > 1:
#        mpi_status = "[rank %d] " % mpi_helper.rank
#    else:
#        mpi_status = ""
#    logger.debug(agw, "%s%s: %s = %.3f G (current usage = %.3f G)",
#            mpi_status, message, shape, np.prod(shape) * 8 / (2**30), lib.current_memory()[0] / 2**10)

def build_se_moments_opt(
        agw, nmom,
        Lpq=None,
        mo_energy=None,
        mo_coeff=None,
        npoints=48,
        ainit=10,
):
    """
    Optimised routine for computing the DD response moments. Assumes
    `Lpq` is precomputed, `rixc` is `None`.
    """

    vlog = NoLogger()

    if mo_energy is None:
        mo_energy = agw._scf.mo_energy

    nmo = agw.nmo
    naux = agw.with_df.get_naoaux()
    naux_block = naux
    nocc = agw.nocc
    nvir = nmo - nocc
    nov = nocc * nvir
    eo = mo_energy[:nocc]
    ev = mo_energy[nocc:]

    # Get 3c integrals
    if Lpq is None:
        Lpq = agw.ao2mo(mo_coeff)
    Lpq = Lpq.reshape(naux, nmo, nmo)

    # Get rotation matrix and A+B
    apb = Lpq[:, :nocc, nocc:].reshape(naux, nov) * np.sqrt(2)
    apb = np.concatenate([apb, apb], axis=1)
    apb = compress_low_rank(apb, apb)

    # Get compressed MP
    d = lib.direct_sum("a-i->ia", ev, eo).ravel()
    d = np.concatenate([d, d])
    l = apb[0] * d[None]
    r = apb[1]
    mp_l, mp_r = compress_low_rank(l, r)

    # Construct inverse
    u = np.dot(apb[1] / d[None], apb[0].T)
    u = np.linalg.inv(np.eye(u.shape[0]) + u)
    u, s, v = np.linalg.svd(u)
    urt_l = u * s[None]**0.5
    urt_r = v * s[:, None]**0.5
    apb_inv_l = np.dot(urt_l.T, apb[0]) / d[None]
    apb_inv_r = np.dot(urt_r, apb[1]) / d[None]
    apb_inv_l, apb_inv_r = compress_low_rank(apb_inv_l, apb_inv_r)

    hole_moms = np.zeros((nmom+1, nmo, nmo))
    part_moms = np.zeros((nmom+1, nmo, nmo))
    for p0, p1 in mpi_helper.prange(0, naux, naux):
        # Get rotation matrix
        rot = Lpq[p0:p1, :nocc, nocc:].reshape(p1-p0, nov)
        rot = np.concatenate([rot, rot], axis=1)

        # Perform the offset integral
        offset = momzero_NI.MomzeroOffsetCalcGaussLag(d, mp_l, mp_r, rot, npoints, vlog)
        estval, offset_err = offset.kernel()
        integral_offset = rot * d[None] + estval

        # Perform the rest of the integral
        worker = momzero_NI.MomzeroDeductHigherOrder(d, mp_l, mp_r, rot, npoints, vlog)
        a = worker.opt_quadrature_diag(ainit)
        quad = worker.get_quad(a)
        integral = np.zeros((p1-p0, nov*2))
        integral_h = np.zeros((p1-p0, nov*2))
        integral_q = np.zeros((p1-p0, nov*2))
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
        pinv_norm = np.sum(d ** -2)
        pinv_norm += 2.0 * apb_inv_l * apb_inv_r / d[None]
        pinv_norm += np.linalg.norm((apb_inv_l, apb_inv_r)) ** 4
        pinv_norm **= 0.5
        t0_err = err * pinv_norm
        # self.check_errors(t0_err, rot.size)
        # self.test_eta0_error(t0, rot, apb, amb)

        # Get the first order moment
        t1 = rot * d[None]

        # Populate the moments
        moments = np.zeros((nmom+1, *t0.shape))
        moments[0] = t0
        moments[1] = t1
        for i in range(2, nmom+1):
            moments[i] = moments[i-2] * d[None]**2
            moments[i] += np.linalg.multi_dot((moments[i-2], mp_r.T, mp_l))  # NOTE needs the full mp

        for q0, q1 in lib.prange(0, naux, 500):
            # Rotate right side
            rotq = Lpq[q0:q1, :nocc, nocc:].reshape(q1-q0, nov)
            rotq = np.concatenate([rotq, rotq], axis=1)
            tild_etas = lib.einsum("nPk,Qk->nPQ", moments, rotq)  # NOTE likely RAM bottleneck

            # Construct the SE moments
            if agw.diag_sigma:
                tild_sigma = np.zeros((nmo, nmom+1, nmo))
                for x in range(nmo):
                    Lpx = Lpq[p0:p1, :, x]
                    Lqx = Lpq[q0:q1, :, x]
                    tild_sigma[x] = lib.einsum("Pp,Qp,nPQ->np", Lpx, Lqx, tild_etas)
                moms = np.arange(nmom+1)
                for n in range(nmom+1):
                    fp = scipy.special.binom(n, moms)
                    fh = fp * (-1)**moms
                    eon = np.power.outer(mo_energy[:agw.nocc], n-moms)
                    evn = np.power.outer(mo_energy[agw.nocc:], n-moms)
                    th = lib.einsum("t,kt,ktp->p", fh, eon, tild_sigma[:agw.nocc])
                    tp = lib.einsum("t,ct,ctp->p", fp, evn, tild_sigma[agw.nocc:])
                    hole_moms[n] += np.diag(th)
                    part_moms[n] += np.diag(tp)
            else:
                tild_sigma = np.zeros((nmo, nmom+1, nmo, nmo))
                moms = np.arange(nmom+1)
                for x in range(nmo):
                    Lpx = Lpq[p0:p1, :, x]
                    Lqx = Lpq[q0:q1, :, x]
                    tild_sigma[x] = lib.einsum("Pp,Qq,nPQ->npq", Lpx, Lqx, tild_etas)
                for n in range(nmom+1):
                    fp = scipy.special.binom(n, moms)
                    fh = fp * (-1)**moms
                    eon = np.power.outer(mo_energy[:agw.nocc], n-moms)
                    evn = np.power.outer(mo_energy[agw.nocc:], n-moms)
                    th = lib.einsum("t,kt,ktpq->pq", fh, eon, tild_sigma[:agw.nocc])
                    tp = lib.einsum("t,ct,ctpq->pq", fp, evn, tild_sigma[agw.nocc:])
                    hole_moms[n] += th
                    part_moms[n] += tp

    mpi_helper.barrier()
    hole_moms = mpi_helper.allreduce(hole_moms)
    part_moms = mpi_helper.allreduce(part_moms)

    hole_moms = 0.5 * (hole_moms + hole_moms.swapaxes(1, 2))
    part_moms = 0.5 * (part_moms + part_moms.swapaxes(1, 2))

    return hole_moms, part_moms
