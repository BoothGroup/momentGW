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


def compress_low_rank(ri_l, ri_r, tol=1e-10, return_rot=False):
    """Perform the low-rank compression."""

    naux_init = ri_l.shape[0]

    u, s, v = np.linalg.svd(ri_l, full_matrices=False)
    nwant = sum(s > tol)
    rot = u[:, :nwant]
    ri_l = np.dot(rot.T, ri_l)
    ri_r = np.dot(rot.T, ri_r)

    rot_full = rot

    u, s, v = np.linalg.svd(ri_r, full_matrices=False)
    nwant = sum(s > tol)
    rot = u[:, :nwant]
    ri_l = np.dot(rot.T, ri_l)
    ri_r = np.dot(rot.T, ri_r)

    rot_full = np.dot(rot_full, rot)

    if return_rot:
        return ri_l, ri_r, rot_full
    else:
        return ri_l, ri_r


def compress_low_rank_symmetric(ri, tol=1e-10, return_rot=False):
    """Perform the low-rank compression."""

    naux_init = ri.shape[0]

    u, s, v = np.linalg.svd(ri, full_matrices=False)
    nwant = sum(s > tol)
    rot = u[:, :nwant]
    ri = np.dot(rot.T, ri)

    rot_full = rot

    u, s, v = np.linalg.svd(ri, full_matrices=False)
    nwant = sum(s > tol)
    rot = u[:, :nwant]
    ri = np.dot(rot.T, ri)

    rot_full = np.dot(rot_full, rot)

    if return_rot:
        return ri, rot_full
    else:
        return ri


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
    Lpq : numpy.ndarray
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
    compress=1,
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

    lib.logger.debug(gw, "Constructing RPA moments:")
    memory_string = lambda: "Memory usage: %.2f GB" % (lib.current_memory()[0] / 1e3)

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

    eo = mo_energy_w[mo_occ_w > 0]
    ev = mo_energy_w[mo_occ_w == 0]
    naux = Lpq.shape[0]
    nov = eo.size * ev.size

    # Get 3c integrals
    if Lia is None:
        Lia = Lpq[:, mo_occ_w > 0][:, :, mo_occ_w == 0]
    Lia = Lia.reshape(naux, nov)

    # Compress the 3c integrals
    lib.logger.debug(gw, "  Compressing 3c integrals")
    Lia, aux_rot = compress_low_rank_symmetric(Lia, return_rot=True)
    Lpq = lib.einsum("Lpq,LQ->Qpq", Lpq, aux_rot)
    naux_comp = Lia.shape[0]
    lib.logger.debug(gw, "  Size of compressed auxiliaries: %s", naux_comp)
    lib.logger.debug(gw, "  %s", memory_string())

    # Construct intermediates
    d = lib.direct_sum("a-i->ia", ev, eo).ravel()
    Lia_d = Lia * d[None]
    Lia_dinv = Lia * d[None]**-1

    # Construct inverse of A-B
    lib.logger.debug(gw, "  Constructing (A-B)^{-1} intermediate")
    u = np.dot(Lia_dinv, Lia.T) * 4
    u = np.linalg.inv(np.eye(u.shape[0]) + u)
    lib.logger.debug(gw, "  %s", memory_string())

    # Perform the offset integral
    offset = momzero_NI.MomzeroOffsetCalcGaussLag(d, Lia_d, Lia, Lia, gw.npoints, vlog)
    estval, offset_err = offset.kernel()
    integral_offset = Lia_d + estval * 4
    lib.logger.debug(gw, "  Offset integral error: %s", offset_err)

    # Get the quadrature for the rest of the integral
    worker = momzero_NI.MomzeroDeductHigherOrder(d, Lia_d, Lia, Lia, gw.npoints, vlog)
    a = worker.opt_quadrature_diag(ainit)
    quad = worker.get_quad(a)
    lib.logger.debug(gw, "  Optimal quadrature parameter: %s", a)

    # Perform the rest of the integral
    integral = np.zeros((naux_comp, nov))
    integral_h = np.zeros((naux_comp, nov))
    integral_q = np.zeros((naux_comp, nov))
    for i, (point, weight) in enumerate(zip(*quad)):
        f = 1.0 / (d**2 + point**2)
        q = np.dot(Lia * f[None], Lia_d.T) * 4
        val_aux = np.linalg.inv(np.eye(q.shape[0]) + q) - np.eye(q.shape[0])
        contrib = np.linalg.multi_dot((q, val_aux, Lia))
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
    lib.logger.debug(gw, "  One-quarter quadrature error: %s", a)
    lib.logger.debug(gw, "  One-half quadrature error: %s", b)
    lib.logger.debug(gw, "  Error estimate: %s", err)

    # Get the zeroth order moment
    integral += integral_offset
    moments = np.zeros((nmom_max + 1, naux_comp, nov))
    moments[0] = integral / d[None]
    moments[0] -= np.linalg.multi_dot((integral, Lia_dinv.T, u, Lia_dinv)) * 4

    # Get the first order moment
    moments[1] = Lia_d

    # Recursively compute the higher-order moments
    for i in range(2, nmom_max + 1):
        moments[i] = moments[i - 2] * d[None] ** 2
        moments[i] += np.linalg.multi_dot(
            (moments[i - 2], Lia.T, Lia_d)
        ) * 4

    # Setup dependent on diagonal SE
    if gw.diagonal_se:
        pq = p = q = "p"
        tild_sigma = np.zeros((mo_energy_g.size, nmom_max + 1, nmo))
        fproc = lambda x: np.diag(x)
    else:
        pq, p, q = "pq", "p", "q"
        tild_sigma = np.zeros((mo_energy_g.size, nmom_max + 1, nmo, nmo))
        fproc = lambda x: x

    # Get the moments in the (aux|aux) basis and rotate to the (mo|mo) basis
    for n in range(nmom_max + 1):
        # Rotate right side
        tild_etas_n = lib.einsum("Pk,Qk->PQ", moments[n], Lia)

        # Construct the moments in the (aux|aux) basis
        for x in range(mo_energy_g.size):
            Lpx = Lpq[:, :, x]
            Lqx = Lpq[:, :, x]
            tild_sigma[x, n] = lib.einsum(f"P{p},Q{q},PQ->{pq}", Lpx, Lqx, tild_etas_n) * 2

    # Construct the SE moments
    hole_moms = np.zeros((nmom_max + 1, nmo, nmo))
    part_moms = np.zeros((nmom_max + 1, nmo, nmo))
    moms = np.arange(nmom_max + 1)
    for n in moms:
        fp = scipy.special.binom(n, moms)
        fh = fp * (-1) ** moms
        eon = np.power.outer(mo_energy_g[mo_occ_g > 0], n - moms)
        evn = np.power.outer(mo_energy_g[mo_occ_g == 0], n - moms)
        th = lib.einsum(f"t,kt,kt{pq}->{pq}", fh, eon, tild_sigma[mo_occ_g > 0])
        tp = lib.einsum(f"t,ct,ct{pq}->{pq}", fp, evn, tild_sigma[mo_occ_g == 0])
        hole_moms[n] += fproc(th)
        part_moms[n] += fproc(tp)

    #mpi_helper.barrier()
    #hole_moms = mpi_helper.allreduce(hole_moms)
    #part_moms = mpi_helper.allreduce(part_moms)

    hole_moms = 0.5 * (hole_moms + hole_moms.swapaxes(1, 2))
    part_moms = 0.5 * (part_moms + part_moms.swapaxes(1, 2))

    return hole_moms, part_moms
