"""Energy functionals."""

import numpy as np
from pyscf import lib

from momentGW import util


def hartree_fock(rdm1, fock, h1e):
    """Hartree--Fock energy functional.

    Parameters
    ----------
    rdm1 : numpy.ndarray
        One-particle reduced density matrix.
    fock : numpy.ndarray
        Fock matrix.
    h1e : numpy.ndarray
        One-electron Hamiltonian.

    Returns:
    -------
    e_1b : float
        Hartree--Fock energy.
    """
    return util.einsum("ij,ji->", rdm1, h1e + fock) * 0.5


def galitskii_migdal(gf, se, flip=False):
    r"""Galitskii--Migdal energy functional.

    Parameters
    ----------
    gf : dyson.Lehmann
        Green's function object.
    se : dyson.Lehmann
        Self-energy object.
    flip : bool, optional
        Default option is to use the occupied Green's function and the
        virtual self-energy. If `flip=True`, the virtual Green's
        function and the occupied self-energy are used instead. Default
        value is `False`.

    Returns:
    -------
    e_2b : float
        Galitskii--Migdal energy.

    Notes:
    -----
    This functional is the analytically integrated version of [1]_

    .. math::
        \frac{\pi}{4} \int d\omega \mathrm{Tr}[G(i\omega)
        \Sigma(i\omega)]

    in terms of the poles of the Green's function and the self-energy.
    This scales as :math:`\mathcal{O}(N^4)` with system size [2]_.

    References:
    ----------
    .. [1] V. M. Galitskii and A. B. Migdal, Sov. Phys. JETP 7, 96,
        1958.
    .. [2] O. J. Backhouse, M. Nusspickel, and G. H. Booth, J. Chem.
        Theory Comput. 16, 2, 2020.
    """

    # Get the correct Green's function and self-energy sectors
    if flip:
        gf = gf.virtual()
        se = se.occupied()
    else:
        gf = gf.occupied()
        se = se.virtual()

    # Compute the Galitskii--Migdal energy in blocks
    e_2b = 0.0
    for p0, p1 in lib.prange(0, se.naux, 256):
        vu = util.einsum("pk,px->kx", se.couplings[:, p0:p1], gf.couplings)
        denom = lib.direct_sum("x-k->kx", gf.energies, se.energies[p0:p1])
        e_2b += np.ravel(util.einsum("kx,kx,kx->", vu, vu.conj(), 1.0 / denom))[0]

    # Apply the factor 2
    e_2b *= 2.0

    return e_2b


def galitskii_migdal_g0(mo_energy, mo_occ, se, flip=False):
    r"""Galitskii--Migdal energy functional for the non-interacting Green's
    function.

    Parameters
    ----------
    mo_energy : numpy.ndarray
        MO energies (poles of the Green's function).
    mo_occ : numpy.ndarray
        MO occupancies.
    se : dyson.Lehmann
        Self-energy.
    flip : bool, optional
        Default option is to use the occupied Green's function and the
        virtual self-energy. If `flip=True`, the virtual Green's
        function and the occupied self-energy are used instead. Default
        value is `False`.

    Returns:
    -------
    e_2b : float
        Galitskii--Migdal energy.

    Notes:
    -----
    This functional is the analytically integrated version of [1]_

    .. math::
        \frac{\pi}{4} \int d\omega \\
            \mathrm{Tr}[G_{0}(i\omega) \Sigma(i\omega)]

    in terms of the poles of the mean-field Green's function and the
    self-energy. This scales as :math:`\mathcal{O}(N^3)` with system
    size [2]_.

    References:
    ----------
    .. [1] V. M. Galitskii and A. B. Migdal, Sov. Phys. JETP 7, 96,
        1958.
    .. [2] O. J. Backhouse, M. Nusspickel, and G. H. Booth, J. Chem.
        Theory Comput. 16, 2, 2020.
    """

    # Get the correct Green's function and self-energy sectors
    if flip:
        mo = mo_energy[mo_occ == 0]
        se = se.occupied()
        se.couplings = se.couplings[mo_occ == 0]
    else:
        mo = mo_energy[mo_occ > 0]
        se = se.virtual()
        se.couplings = se.couplings[mo_occ > 0]

    # Compute the Galitskii--Migdal energy in blocks
    denom = lib.direct_sum("i-j->ij", mo, se.energies)
    e_2b = np.ravel(util.einsum("xk,xk,xk->", se.couplings, se.couplings.conj(), 1.0 / denom))[0]

    # Apply the factor 2
    e_2b *= 2.0

    return e_2b
