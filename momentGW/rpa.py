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


class NIException(RuntimeError):
    pass


def compress_low_rank(ri_l, ri_r, tol=1e-10):
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


def compress_low_rank_symmetric(ri, tol=1e-10):
    """Perform the low-rank compression."""

    naux_init = ri.shape[0]

    u, s, v = np.linalg.svd(ri, full_matrices=False)
    nwant = sum(s > tol)
    rot = u[:, :nwant]
    ri = np.dot(rot.T, ri)

    u, s, v = np.linalg.svd(ri, full_matrices=False)
    nwant = sum(s > tol)
    rot = u[:, :nwant]
    ri = np.dot(rot.T, ri)

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

    lib.logger.debug(gw, "Constructing RPA moments")

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

    import logging

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

    # Get rotation matrix and A+B - factor is carried
    lib.logger.debug(gw, "Constructing (A+B)")
    rot = apb = Lia.reshape(naux, nov)
    if compress > 3:
        apb = compress_low_rank_symmetric(apb)
        lib.logger.debug(gw, "Compressed (A+B) from %s -> %s", (naux, nov), apb.shape)

    # Get compressed MP
    lib.logger.debug(gw, "Constructing S_L and S_R")
    d = lib.direct_sum("a-i->ia", ev, eo).ravel()
    s_l = apb * d[None]
    s_r = apb
    if compress > 0:
        s_l, s_r = compress_low_rank(s_l, s_r)
        lib.logger.debug(gw, "Compressed S_L from %s -> %s", apb.shape, s_l.shape)
        lib.logger.debug(gw, "Compressed S_R from %s -> %s", apb.shape, s_r.shape)

    # Construct inverse of A-B - RAM bottleneck
    lib.logger.debug(gw, "Constructive (A-B)^{-1}")
    u = np.dot(apb / d[None], apb.T) * 4
    u = np.linalg.inv(np.eye(u.shape[0]) + u)
    w, u = np.linalg.eigh(u)
    u = u * w[None] ** 0.5
    apb_inv = np.linalg.multi_dot((u.T, apb)) / d[None]
    # Moved this deletion later.
    # del apb, u
    if compress > 5:
        shape = apb_inv.shape
        apb_inv = compress_low_rank_symmetric(apb_inv)
        lib.logger.debug(gw, "Compressed (A-B)^{-1} from %s -> %s", shape, apb_inv.shape)

    # Perform the offset integral and set up optimised quadrature grid.

    # Old code using Vayesta.
    # offset = momzero_NI.MomzeroOffsetCalcGaussLag(d, s_l, s_r, rot, gw.npoints, vlog)
    # estval, offset_err = offset.kernel()
    # estval *= 4  # For restricted symmetry and carried factor
    # integral_offset = rot * d[None] + estval

    # worker = momzero_NI.MomzeroDeductHigherOrder(d, s_l, s_r, rot, gw.npoints, vlog)
    # a = worker.opt_quadrature_diag(ainit)
    # quad = worker.get_quad(a)

    # New code purely in momentGW.
    # Thoughts about parallelising this:
    #   - I've kept everything dependent only on the 3c eris, rather than the previous RI objects, for simplicity.
    #   - the offset integral has the same considerations as the main one (same cost, similar approach with distributed
    #     eris).
    #   - Given the numerical approaches in the quadratures (both in optimising the quadrature and solving the required
    #     equations to construct the Gauss-Laguerre quadrature), we probably need to only actually solve this on one
    #     process and broadcast the result to the others to avoid all sorts of horrors.
    #   - The diagonal evaluations required for the quadrature optimisation are straightforward. All their
    #     dependencies on the eris are in the intermediate below (diag_eri) which is the same size as D, so we could
    #     either construct this ahead of time (as currently implemented) or actually parallelise all diagonal
    #     approximation evaluations. However, once this intermediate is constructed all other operations scale as O(ov)
    #     so it won't be a bottleneck.

    # Intermediate for quadrature optimisations; same dimension as d.
    diag_eri = np.einsum("np,np->p", apb, apb)  # O(n_aux ov)

    offset_quad = optimise_offset_quad(gw.npoints, d, diag_eri, vlog)
    integral_offset = rot * d[None] + 4 * eval_offset_integral(d, apb, offset_quad)

    # Get the quadrature for the rest of the integral
    quad = optimise_main_quad(gw.npoints, d, diag_eri, vlog)

    del apb, u

    hole_moms = np.zeros((nmom_max + 1, nmo, nmo))
    part_moms = np.zeros((nmom_max + 1, nmo, nmo))
    for p0, p1 in lib.prange(0, naux, 100):
        # Perform the rest of the integral
        integral = np.zeros((p1 - p0, nov))
        integral_h = np.zeros((p1 - p0, nov))
        integral_q = np.zeros((p1 - p0, nov))
        for i, (point, weight) in enumerate(zip(*quad)):
            f = 1.0 / (d ** 2 + point ** 2)
            q = np.dot(s_r * f[None], s_l.T) * 4  # NOTE CPU bottleneck
            lrot = rot[p0:p1] * f[None]
            val_aux = np.linalg.inv(np.eye(q.shape[0]) + q) - np.eye(q.shape[0])
            contrib = np.linalg.multi_dot((lrot, s_l.T, val_aux, s_r)) * 4
            contrib *= f[None]
            contrib *= point ** 2
            contrib /= np.pi
            integral += weight * contrib
            if i % 2 == 0:
                integral_h += 2 * weight * contrib
            if i % 4 == 0:
                integral_q += 4 * weight * contrib
        a = np.linalg.norm(integral_q - integral)
        b = np.linalg.norm(integral_h - integral)
        err = estimate_error_clencur(a, b)

        # Get the zeroth order moment
        integral_part = integral + integral_offset[p0:p1]
        t0 = integral_part / d[None]
        t0 -= np.linalg.multi_dot((integral_part, apb_inv.T, apb_inv)) * 4

        # Get the errors
        pinv_norm = np.sum(d ** -2)
        pinv_norm += 8.0 * apb_inv * apb_inv / d[None]
        pinv_norm += (4.0 * np.linalg.norm((apb_inv, apb_inv))) ** 4
        pinv_norm **= 0.5
        t0_err = err * pinv_norm
        # self.check_errors(t0_err, rot.size)
        # self.test_eta0_error(t0, rot, apb, amb)

        # Get the first order moment
        t1 = rot[p0:p1] * d[None]

        # Populate the moments
        moments = np.zeros((nmom_max + 1, t0.shape[0], nov))
        moments[0] = t0
        moments[1] = t1
        for i in range(2, nmom_max + 1):
            moments[i] = moments[i - 2] * d[None] ** 2
            moments[i] += np.linalg.multi_dot(
                (moments[i - 2], s_r.T, s_l)
            ) * 4

        # Rotate right side
        tild_etas = lib.einsum("nPk,Qk->nPQ", moments, rot)  # NOTE likely RAM bottleneck

        # Setup dependent on diagonal SE
        if gw.diagonal_se:
            pq = p = q = "p"
            tild_sigma = np.zeros((mo_energy_g.size, nmom_max + 1, nmo))
            fproc = lambda x: np.diag(x)
        else:
            pq, p, q = "pq", "p", "q"
            tild_sigma = np.zeros((mo_energy_g.size, nmom_max + 1, nmo, nmo))
            fproc = lambda x: x

        # Construct the moments in the (aux|aux) basis
        for x in range(mo_energy_g.size):
            Lpx = Lpq[p0:p1, :, x]
            Lqx = Lpq[:, :, x]
            tild_sigma[x] = lib.einsum(f"P{p},Q{q},nPQ->n{pq}", Lpx, Lqx, tild_etas) * 2

        # Construct the SE moments
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

    # mpi_helper.barrier()
    # hole_moms = mpi_helper.allreduce(hole_moms)
    # part_moms = mpi_helper.allreduce(part_moms)

    hole_moms = 0.5 * (hole_moms + hole_moms.swapaxes(1, 2))
    part_moms = 0.5 * (part_moms + part_moms.swapaxes(1, 2))

    return hole_moms, part_moms


# Evaluation of the offset integral (could also move main integral to similar function).


def eval_offset_integral(d, apb, quad):
    """Evaluate the offset integral resulting from exact integration of higher-order terms within the main integral into
    an exponentially decaying form.
    Parameters
    ----------
    d : np.ndarray
         array of orbital energy differences.
    apb : np.ndarray
        low-rank RI contribution to A+B
    quad : tuple of np.ndarray
        quadrature points and weights to evaluate integral at.

    Returns
    -------
    integral : np.ndarray
        Estimated value of the integral from numerical integration.
    """

    integral = np.zeros_like(apb)

    for point, weight in zip(*quad):
        expval = np.exp(-point * d)

        lhs = np.einsum("np,p,p,mp->nm", apb, expval, d, apb)  # O(N_aux^2 ov)
        rhs = np.einsum("np,p->np", apb, expval)  # O(N_aux ov)

        res = np.dot(lhs, rhs)
        integral += res * weight  # O(N_aux^2 ov)
        # integral += np.dot(lhs, rhs) * weight  # O(N_aux^2 ov)

    return integral


# Optimisation of different quadratures.
# Note that since diagonal approximation has no coupling between spin channels we can just optimise in single spin
# channel using spatial quantities.


def optimise_main_quad(npoints, d, diag_eri, vlog):
    """Optimise grid spacing of clenshaw-curtis quadrature for main integral.
    All input parameters are spatial/in a single spin channel, eg. (aa|aa) .
    Parameters
    ----------
    npoints : int
        Number of points in the quadrature grid.
    d : np.ndarray
        Diagonal array of orbital energy differences.
    diag_ri : np.ndarray
        Diagonal of the ri contribution to (A-B)(A+B).
    Returns
    -------
    quad: tuple
        Tuple of (points, weights) for the quadrature.
    """

    bare_quad = gen_clencur_quad_inf(npoints, even=True)
    # Get exact integral.
    exact = np.sum((d ** 2 + np.multiply(d, diag_eri)) ** (0.5)) - sum(d) - 0.5 * np.dot(d ** (-1),
                                                                                         np.multiply(d, diag_eri))

    def integrand(quad):
        return eval_diag_main_integral(d, diag_eri, quad)

    return get_optimal_quad(bare_quad, integrand, exact, vlog)


def optimise_offset_quad(npoints, d, diag_eri, vlog):
    """Optimise grid spacing of Gauss-Laguerre quadrature for main integral.
    Parameters
    ----------
    npoints : int
        Number of points in the quadrature grid.
    d : np.ndarray
        Diagonal array of orbital energy differences.
    diag_ri : np.ndarray
        Diagonal of the ri contribution to (A-B)(A+B).
    Returns
    -------
    quad: tuple
        Tuple of (points, weights) for the quadrature.
    """
    bare_quad = gen_gausslag_quad_semiinf(npoints)
    # Get exact integral.
    exact = 0.5 * np.dot(d ** (-1), np.multiply(d, diag_eri))

    def integrand(quad):
        return eval_diag_offset_integral(d, diag_eri, quad)

    return get_optimal_quad(bare_quad, integrand, exact, vlog)


def get_optimal_quad(bare_quad, integrand, exact, vlog):
    def compute_diag_err(spacing):
        """Compute the error in the diagonal integral."""
        quad = rescale_quad(10 ** spacing, bare_quad)
        integral = integrand(quad)
        return abs(integral - exact)

    # Get the optimal spacing, optimising exponent for stability.
    res = scipy.optimize.minimize_scalar(compute_diag_err, bounds=(-6, 2), method="bounded")
    if not res.success:
        raise NIException("Could not optimise `a' value.")
    solve = 10 ** res.x
    # Debug message once we get logging sorted.
    # ("Used minimisation to optimise quadrature grid: a= %.2e  penalty value= %.2e (smaller is better)"
    # % (solve, res.fun))
    return rescale_quad(solve, bare_quad)


def eval_diag_main_integral(d, diag_eri, quad):
    def diag_contrib(diag_mat, freq):
        return (np.full_like(diag_mat, fill_value=1.0) - freq ** 2 * (diag_mat + freq ** 2) ** (-1)) / np.pi

    # Intermediates requiring contributions from distributed ERIs.
    # These all have size ov.
    diagmat1 = d ** 2 + np.multiply(d, diag_eri)
    diagmat2 = d ** 2

    integral = 0.0

    for (point, weight) in zip(*quad):
        f = (d ** 2 + point ** 2) ** (-1)

        integral += weight * (
                    sum(diag_contrib(diagmat1, point)  # Integral for diagonal approximation to ((A-B)(A+B))^(0.5)
                        - diag_contrib(diagmat2, point))  # Equivalent integral for (D^2)^(0.5) (deduct)
                    - point ** 2 * np.dot(f ** 2, np.multiply(d, diag_eri)) / np.pi)  # Higher order terms from offset.
    return integral


def eval_diag_offset_integral(d, diag_eri, quad):
    integral = 0.0

    for point, weight in zip(*quad):
        integral += weight * np.dot(np.exp(- 2 * point * d), np.multiply(d, diag_eri))
    return integral


# Functions to generate quadrature points and weights
def rescale_quad(a, bare_quad):
    """Rescale quadrature for grid spacing `a`."""
    return bare_quad[0] * a, bare_quad[1] * a


def gen_clencur_quad_inf(npoints, even=False):
    """Generate quadrature points and weights for Clenshaw-Curtis quadrature over infinite range (-inf to +inf)"""
    symfac = 1.0 + even
    # If even we only want points up to t <= pi/2
    tvals = [(j / npoints) * (np.pi / symfac) for j in range(1, npoints + 1)]

    points = [1.0 / np.tan(t) for t in tvals]
    weights = [np.pi * symfac / (2 * npoints * (np.sin(t) ** 2)) for t in tvals]
    if even:
        weights[-1] /= 2
    return np.array(points), np.array(weights)


def gen_gausslag_quad_semiinf(npoints):
    p, w = np.polynomial.laguerre.laggauss(npoints)
    # Additional weighting accounting for multiplication of the function by e^{x}e^{-x} in order to apply Gauss-Laguerre
    # quadrature. Rescaling then results in simply multiplying both the points and weights by the grid spacing.
    # Technically what we're doing is taking our original integral
    #       \int_{0}^{\inf} f(x) dx = \int_{0}^{\inf} e^{-x} f(x) e^{x} dx
    #                               = \int_{0}^{\inf} e^{-x} g(x) dx
    # so the final form is suitable for Guass-Laguerre quadrature.
    # Applying a grid spacing of a, this can equivalently be thought of as changing the exponent of this exponential
    # then rescaling the integration variable.

    w = w * np.exp(p)
    return np.array(p), np.array(w)


def estimate_error_clencur(a, b):
    if a - b < 1e-10:
        #log.info("RIRPA error numerically zero.")
        return 0.0
    # This is Eq. 103 from https://arxiv.org/abs/2301.09107
    roots = np.roots([1, 0, a / (a - b), -b / (a - b)])
    # From physical considerations require real root between zero and one, since this is value of e^{-\beta n_p}.
    # If there are multiple (if this is even possible) we take the largest.
    real_roots = roots[abs(roots.imag) < 1e-10].real
    # Warnings to add with logging...
    #if len(real_roots) > 1:
        #log.warning(
        #    "Nested quadrature error estimation gives %d real roots. Taking smallest positive root.",
        #    len(real_roots),
        #)
    #else:
        #log.debug(
        #    "Nested quadrature error estimation gives %d real root.",
        #    len(real_roots),
        #)

    if not (any((real_roots > 0) & (real_roots < 1))):
        #log.critical(
        #    "No real root found between 0 and 1 in NI error estimation; returning nan."
        #)
        return np.nan
    else:
        # This defines the values of e^{-\beta n_p}, where we seek the value of \alpha e^{-4 \beta n_p}
        wanted_root = real_roots[real_roots > 0.0].min()
    # We could go via the steps
    #   exp_beta_4n = wanted_root ** 4
    #   alpha = a * (exp_beta_n + exp_beta_4n**(1/4))**(-1)
    # But instead go straight for
    error = b / (1 + wanted_root ** (-2.0))
    return error