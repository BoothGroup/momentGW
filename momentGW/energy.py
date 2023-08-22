"""
Energy functionals.
"""

import numpy as np
from pyscf import lib


def hartree_fock(rdm1, fock, h1e):
    """Hartree--Fock energy functional."""
    return lib.einsum("ij,ji->", rdm1, h1e + fock) * 0.5


def galitskii_migdal(gf, se, flip=False):
    r"""Galitskii--Migdal energy functional.

    Parameters
    ----------
    gf : GreensFunction
        Green's function.
    se : SelfEnergy
        Self-energy.
    flip : bool, optional
        Default option is to use the occupied Green's function and the
        virtual self-energy. If `flip=True`, the virtual Green's
        function and the occupied self-energy are used instead. Default
        value is `False`.

    Returns
    -------
    e_2b : float
        Galitskii--Migdal energy.

    Notes
    -----
    This functional is the analytically integrated version of

    .. math:: \frac{\pi}{4} \int d\omega \mathrm{Tr}[G(i\omega) \Sigma(i\omega)]

    in terms of the poles of the Green's function and the self-energy.
    This scales as :math:`\mathcal{O}(N^4)` with system size.
    """

    if flip:
        gf = gf.get_virtual()
        se = se.get_occupied()
    else:
        gf = gf.get_occupied()
        se = se.get_virtual()

    e_2b = 0.0
    for i in range(gf.naux):
        v_gf = gf.coupling[:, i]
        v_se = se.coupling
        v = v_se * v_gf[:, None]
        denom = gf.energy[i] - se.energy

        e_2b += np.ravel(lib.einsum("xk,yk,k->", v, v.conj(), 1.0 / denom))[0]

    e_2b *= 2.0

    return e_2b


def galitskii_migdal_g0(mo_energy, mo_occ, se, flip=False):
    r"""
    Galitskii--Migdal energy functional for the non-interacting Green's
    function.

    Parameters
    ----------
    mo_energy : numpy.ndarray
        MO energies (poles of the Green's function).
    mo_occ : numpy.ndarray
        MO occupancies.
    se : SelfEnergy
        Self-energy.
    flip : bool, optional
        Default option is to use the occupied Green's function and the
        virtual self-energy. If `flip=True`, the virtual Green's
        function and the occupied self-energy are used instead. Default
        value is `False`.

    Returns
    -------
    e_2b : float
        Galitskii--Migdal energy.

    Notes
    -----
    This functional is the analytically integrated version of

    .. math:: \frac{\pi}{4} \int d\omega \mathrm{Tr}[G_{0}(i\omega) \Sigma(i\omega)]

    in terms of the poles of the mean-field Green's function and the
    self-energy. This scales as :math:`\mathcal{O}(N^3)` with system
    size.
    """

    if flip:
        mo = mo_energy[mo_occ == 0]
        se = se.get_occupied()
        se.coupling = se.coupling[mo_occ == 0]
    else:
        mo = mo_energy[mo_occ > 0]
        se = se.get_virtual()
        se.coupling = se.coupling[mo_occ > 0]

    denom = lib.direct_sum("i-j->ij", mo, se.energy)

    e_2b = np.ravel(lib.einsum("xk,xk,xk->", se.coupling, se.coupling.conj(), 1.0 / denom))[0]
    e_2b *= 2.0

    return e_2b
