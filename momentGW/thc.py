"""
Tensor Hyper Contraction formulation of the moment GW approach
"""

import numpy as np
from momentGW.base import BaseGW
#from momentGW.gw import GW
from pyscf import lib
from pyscf.agf2 import mpi_helper

from momentGW.rpa import compress_eris, gen_gausslag_quad_semiinf, get_optimal_quad


def build_se_thc_moments_drpa(
    gw,
    nmom_max,
    Lpq,
    Lia,
    mo_energy=None,
    mo_occ=None,
    ainit=10,
    compress=True,
):
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

    d_full = lib.direct_sum("a-i->ia", ev, eo).ravel()
    d = d_full[p0:p1]

def optimise_f_quad(npoints, d, z_point):
    """
    Optimise grid spacing of Gauss-Laguerre quadrature for F(z)

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
    exact = np.dot(d, np.multiply(z_point**2, np.eye(len(d))))** (-1)

    def integrand(quad):
        return eval_f_integral(d, z_point, quad)

    return get_optimal_quad(bare_quad, integrand, exact)

def eval_f_integral(d, z_point, quad):
    integral = 0.0

    for point, weight in zip(*quad):
        integral += weight * d * np.dot(np.exp(- 2 * point * d), np.sin(z_point*point)/z_point)
    return integral