"""
Construct RPA moments.
"""
import logging
import numpy as np
import scipy.special
from pyscf import lib
from pyscf.agf2 import mpi_helper
from vayesta.rpa.rirpa.NI_eval import NumericalIntegratorBase

import pickle
def StoreData(data_list: list, name_of_pickle: str):
    """ Stores list of data. Overwrites any previous data in the pickle file. """
    # Delete previous data
    pickle_file = open(name_of_pickle, 'w+')
    pickle_file.truncate(0)
    pickle_file.close()
    # Write new data
    pickle_file = open(name_of_pickle, 'ab')  # Mode: append + binary
    pickle.dump(data_list, pickle_file)
    pickle_file.close()

import momentGW.thc as gwthc
from vayesta.core.vlog import NoLogger, VFileHandler


# TODO silence Vayesta


class NIException(RuntimeError):
    pass


def mpi_dot(x, y):
    """
    Simple MPI dot-product with distribution over the first index of
    the input array `x`.
    """

    size = x.shape[0]
    res = np.zeros((x.shape[0], y.shape[1]))
    for p0, p1 in mpi_helper.prange(0, size, size):
        res[p0:p1] = np.dot(x[p0:p1], y)

    res = mpi_helper.allreduce(res)

    return res


def compress_eris(Lpq, Lia, tol=1e-10):
    """
    Algorithm to compress distributed CDERIs auxiliary index.
    This algorithm requires O(naux^2) memory and O(naux^3 + naux ov nproc^{-1}) time on each process.

    Parameters
    ----------
    Lpq : np.ndarray
        The (full) CDERIs in the AO basis.
    Lia : np.ndarray
        The portion of the MO particle-hole CDERIs stored on this process.
    tol : float
        The tolerance for the eigenvalue truncation. Default is 1e-10.

    Returns
    -------
    Lpq : np.ndarray
        The CDERIs in the AO basis with a compressed auxiliary basis.
    Lia : np.ndarray
        The compressed MO particle-hole CDERIs stored on this process with a compressed auxiliary basis.
    """

    naux_init = Lia.shape[0]

    shape_pq = Lpq.shape[1:]
    shape_ia = Lia.shape[1:]
    Lpq = Lpq.reshape(naux_init, -1)
    Lia = Lia.reshape(naux_init, -1)

    intermed = np.dot(Lia, Lia.T)
    intermed = mpi_helper.reduce(intermed, root=0)
    # Compute on root and broadcast for the sake of avoiding numerical issues.
    if mpi_helper.rank == 0:
        e, v = np.linalg.eigh(intermed)
        want = abs(e) > tol
        rot = v[:, want]
    else:
        rot = np.zeros((0,))
    del intermed

    rot = mpi_helper.bcast(rot, root=0)

    Lia = np.dot(rot.T, Lia)
    Lpq = np.dot(rot.T, Lpq)

    Lpq = Lpq.reshape(-1, *shape_pq)
    Lia = Lia.reshape(-1, *shape_ia)

    return Lpq, Lia


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

    from vayesta.rpa import ssRPA

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
    ppoints,
    Lpq,
    Lia,
    calc_type,
    mo_energy=None,
    mo_occ=None,
    ainit=10,
    compress=True,
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
    Lia : numpy.ndarray
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
    compress : bool, optional
        Whether to compress the ERIs. Default value is `False`.

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

    nmo = gw.nmo
    naux = gw.with_df.get_naoaux()

    eo = mo_energy_w[mo_occ_w > 0]
    ev = mo_energy_w[mo_occ_w == 0]
    naux = Lpq.shape[0]
    nov = eo.size * ev.size

    # MPI
    p0, p1 = list(mpi_helper.prange(0, nov, nov))[0]
    Lia = Lia.reshape(naux, -1)
    nov_block = p1 - p0
    q0, q1 = list(mpi_helper.prange(0, mo_energy_g.size, mo_energy_g.size))[0]
    mo_energy_g = mo_energy_g[q0:q1]
    mo_occ_g = mo_occ_g[q0:q1]

    # Compress the 3c integrals.
    if compress:
        lib.logger.debug(gw, "  Compressing 3c integrals")
        Lpq, Lia = compress_eris(Lpq, Lia)
        lib.logger.debug(gw, "  Size of uncompressed auxiliaries: %s", naux)
        naux = Lia.shape[0]
        lib.logger.debug(gw, "  Size of compressed auxiliaries: %s", naux)
        lib.logger.debug(gw, "  %s", memory_string())

    # Construct intermediates
    d_full = lib.direct_sum("a-i->ia", ev, eo).ravel()
    d = d_full[p0:p1]
    Lia_d = Lia * d[None]
    Lia_dinv = Lia * d[None]**-1


    # Construct the full d and diag_eri on all processes
    diag_eri = np.zeros((nov,))
    diag_eri[p0:p1] = lib.einsum("np,np->p", Lia, Lia)  # O(N_aux ov)
    diag_eri = mpi_helper.allreduce(diag_eri)


    # Perform the offset integral
    offset_quad = optimise_offset_quad(gw.npoints, d_full, diag_eri)
    integral_offset = Lia * d[None] + 4 * eval_offset_integral(d, Lia, offset_quad)

    # Get the quadrature for the rest of the integral
    quad = optimise_main_quad(gw.npoints, d_full, diag_eri)

    # Perform the rest of the integral
    integral = np.zeros((naux, nov_block))
    integral_h = np.zeros((naux, nov_block))
    integral_q = np.zeros((naux, nov_block))
    StoreData(d, 'D_result')

    for i, (point, weight) in enumerate(zip(*quad)):
        if calc_type=='normal':
            f = 1.0 / (d ** 2 + point ** 2)
            q = np.dot(Lia * f[None], Lia_d.T) * 4
        if calc_type=='thc':
            # f = 1.0 / (d ** 2 + point ** 2)
            # q_calc = gwthc.MomzeroOffsetCalcCC(d, Lia,Lia_d, naux, ppoints, point,logging.getLogger(__name__)).kernel()
            # q = q_calc[0]
            # print(q - np.dot(Lia * f[None], Lia_d.T) * 4)

            f_calc = gwthc.MomzeroOffsetCalcCC(d, ppoints, point,
                                                    logging.getLogger(__name__)).kernel()
            f = f_calc[0]
            # print(f-1 / (d ** 2 + point ** 2))
            q = np.dot(Lia * f[None], Lia_d.T) * 4
            # print(q - np.dot(Lia * 1 / (d ** 2 + point ** 2), Lia_d.T) * 4)

        q = mpi_helper.allreduce(q)
        val_aux = np.linalg.inv(np.eye(q.shape[0]) + q) - np.eye(q.shape[0])
        contrib = np.linalg.multi_dot((q, val_aux, Lia))
        del q, val_aux
        contrib *= f[None]
        contrib *= point**2
        contrib /= np.pi
        integral += weight * contrib
        if i % 2 == 0:
            integral_h += 2 * weight * contrib
        if i % 4 == 0:
            integral_q += 4 * weight * contrib
    a, b = mpi_helper.allreduce(np.array([
        sum((integral_q - integral).ravel() ** 2),
        sum((integral_h - integral).ravel() ** 2),
    ]))
    a, b = a ** 0.5, b ** 0.5
    err = estimate_error_clencur(a, b)
    lib.logger.debug(gw, "  One-quarter quadrature error: %s", a)
    lib.logger.debug(gw, "  One-half quadrature error: %s", b)
    lib.logger.debug(gw, "  Error estimate: %s", err)

    # Construct inverse of A-B
    lib.logger.debug(gw, "  Constructing (A-B)^{-1} intermediate")
    u = np.dot(Lia_dinv, Lia.T) * 4
    u = mpi_helper.allreduce(u)
    u = np.linalg.inv(np.eye(u.shape[0]) + u)
    lib.logger.debug(gw, "  %s", memory_string())

    # Get the zeroth order moment
    integral += integral_offset
    moments = np.zeros((nmom_max + 1, naux, nov_block))
    moments[0] = integral / d[None]
    interm = np.linalg.multi_dot((integral, Lia_dinv.T, u))
    interm = mpi_helper.allreduce(interm)
    moments[0] -= np.linalg.multi_dot((interm, Lia_dinv)) * 4
    del u, interm

    # Get the first order moment
    moments[1] = Lia_d

    # Recursively compute the higher-order moments
    for i in range(2, nmom_max + 1):
        moments[i] = moments[i - 2] * d[None] ** 2
        interm = np.dot(moments[i - 2], Lia.T)
        interm = mpi_helper.allreduce(interm)
        moments[i] += np.dot(interm, Lia_d) * 4
        del interm

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
        tild_etas_n = mpi_helper.allreduce(tild_etas_n)  # bad

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
        if np.any(mo_occ_g > 0):
            eon = np.power.outer(mo_energy_g[mo_occ_g > 0], n - moms)
            th = lib.einsum(f"t,kt,kt{pq}->{pq}", fh, eon, tild_sigma[mo_occ_g > 0])
            hole_moms[n] += fproc(th)
        if np.any(mo_occ_g == 0):
            evn = np.power.outer(mo_energy_g[mo_occ_g == 0], n - moms)
            tp = lib.einsum(f"t,ct,ct{pq}->{pq}", fp, evn, tild_sigma[mo_occ_g == 0])
            part_moms[n] += fproc(tp)

    mpi_helper.barrier()
    hole_moms = mpi_helper.allreduce(hole_moms)
    part_moms = mpi_helper.allreduce(part_moms)

    hole_moms = 0.5 * (hole_moms + hole_moms.swapaxes(1, 2))
    part_moms = 0.5 * (part_moms + part_moms.swapaxes(1, 2))

    return hole_moms, part_moms


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

# Evaluation of the offset integral (could also move main integral to similar function).


def eval_offset_integral(d, apb, quad):
    """
    Evaluate the offset integral resulting from exact integration of
    higher-order terms within the main integral into an exponentially
    decaying form.

    Note that the input parameters `d` and `apb` can be distributed
    among processes along their final dimension, and the output
    integral will share this distribution

    Parameters
    ----------
    d : numpy.ndarray
        Array of orbital energy differences.
    apb : numpy.ndarray
        Low-rank RI contribution to A+B
    quad : tuple of numpy.ndarray
        Quadrature points and weights to evaluate integral at.

    Returns
    -------
    integral : numpy.ndarray
        Estimated value of the integral from numerical integration.
    """

    integral = np.zeros_like(apb)

    for point, weight in zip(*quad):
        expval = np.exp(-point * d)

        lhs = np.dot(apb * (expval * d)[None], apb.T)  # O(N_aux^2 ov)
        lhs = mpi_helper.allreduce(lhs)
        rhs = lib.einsum("np,p->np", apb, expval)  # O(N_aux ov)

        res = np.dot(lhs, rhs)
        integral += res * weight  # O(N_aux^2 ov)

    return integral


# Optimisation of different quadratures.
# Note that since diagonal approximation has no coupling between spin channels we can just optimise in single spin
# channel using spatial quantities.


def optimise_main_quad(npoints, d, diag_eri):
    """
    Optimise grid spacing of clenshaw-curtis quadrature for main
    integral. All input parameters are spatial/in a single spin
    channel, eg. (aa|aa).

    Parameters
    ----------
    npoints : int
        Number of points in the quadrature grid.
    d : numpy.ndarray
        Diagonal array of orbital energy differences.
    diag_eri : numpy.ndarray
        Diagonal of the ERI contribution to (A-B)(A+B).

    Returns
    -------
    quad : tuple
        Tuple of (points, weights) for the quadrature.
    """

    bare_quad = gen_clencur_quad_inf(npoints, even=True)
    # Get exact integral.
    exact = (
        + np.sum((d ** 2 + np.multiply(d, diag_eri)) ** 0.5)
        - 0.5 * np.dot(d ** (-1), np.multiply(d, diag_eri))
        - sum(d)
    )

    def integrand(quad):
        return eval_diag_main_integral(d, diag_eri, quad)

    return get_optimal_quad(bare_quad, integrand, exact)


def optimise_offset_quad(npoints, d, diag_eri):
    """
    Optimise grid spacing of Gauss-Laguerre quadrature for main
    integral.

    Parameters
    ----------
    npoints : int
        Number of points in the quadrature grid.
    d : numpy.ndarray
        Diagonal array of orbital energy differences.
    diag_ri : numpy.ndarray
        Diagonal of the ri contribution to (A-B)(A+B).

    Returns
    -------
    quad : tuple
        Tuple of (points, weights) for the quadrature.
    """

    bare_quad = gen_gausslag_quad_semiinf(npoints)
    # Get exact integral.
    exact = 0.5 * np.dot(d ** (-1), np.multiply(d, diag_eri))
    def integrand(quad):
        return eval_diag_offset_integral(d, diag_eri, quad)

    return get_optimal_quad(bare_quad, integrand, exact)


def get_optimal_quad(bare_quad, integrand, exact):
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









