"""
Fock matrix and static self-energy parts with periodic boundary
conditions.
"""

import numpy as np
import scipy.optimize
from dyson import Lehmann
from pyscf import lib

from momentGW import logging, mpi_helper, util
from momentGW.fock import ChemicalPotentialError, FockLoop


# TODO inherit
def _gradient(x, se, fock, nelec, occupancy=2, buf=None):
    """Gradient of the number of electrons w.r.t shift in auxiliary
    energies.
    """
    # TODO buf

    ws, vs = zip(*[s.diagonalise_matrix(f, chempot=x) for s, f in zip(se, fock)])
    chempot, error = search_chempot(ws, vs, se[0].nphys, nelec, occupancy=occupancy)

    nmo = se[0].nphys

    ddm = 0.0
    for i in mpi_helper.nrange(len(ws)):
        w, v = ws[i], vs[i]
        nmo = se[i].nphys
        nocc = np.sum(w < chempot)
        h1 = -np.dot(v[nmo:, nocc:].T.conj(), v[nmo:, :nocc])
        zai = -h1 / lib.direct_sum("i-a->ai", w[:nocc], w[nocc:])
        ddm += util.einsum("ai,pa,pi->", zai, v[:nmo, nocc:], v[:nmo, :nocc].conj()).real * 4

    ddm = mpi_helper.allreduce(ddm)
    grad = occupancy * error * ddm

    return error**2, grad


def search_chempot_constrained(w, v, nphys, nelec, occupancy=2):
    """
    Search for a chemical potential, constraining the k-point
    dependent occupancy to ensure no crossover of states. If this
    is not possible, a ValueError will be raised.
    """

    if nelec == 0:
        return min(wk[0] for wk in w) - 1e-6, 0.0

    nmo = max(len(x) for x in w)
    nkpts = len(w)
    sum0 = sum1 = 0.0

    for i in range(nmo):
        n = 0
        for k in range(nkpts):
            n += np.dot(v[k][:nphys, i].conj().T, v[k][:nphys, i]).real
        n *= occupancy
        sum0, sum1 = sum1, sum1 + n

        if i > 0 and sum0 <= nelec and nelec <= sum1:
            break

    if abs(sum0 - nelec) < abs(sum1 - nelec):
        homo = i - 1
        error = nelec - sum0
    else:
        homo = i
        error = nelec - sum1

    lumo = homo + 1

    if lumo == nmo:
        chempot = np.max(w) + 1e-6
    else:
        e_homo = np.max([x[homo] for x in w])
        e_lumo = np.min([x[lumo] for x in w])

        if e_homo > e_lumo:
            raise ChemicalPotentialError(
                "Could not find a chemical potential under "
                "the constrain of equal k-point occupancy."
            )

        chempot = 0.5 * (e_homo + e_lumo)

    return chempot, error


def search_chempot_unconstrained(w, v, nphys, nelec, occupancy=2):
    """
    Search for a chemical potential, without constraining the
    k-point dependent occupancy.
    """

    w = np.concatenate(w)
    v = np.hstack([vk[:nphys] for vk in v])

    mask = np.argsort(w)
    w = w[mask]
    v = v[:, mask]

    nmo = v.shape[-1]
    sum0 = sum1 = 0.0

    for i in range(nmo):
        n = occupancy * np.dot(v[:nphys, i].conj().T, v[:nphys, i]).real
        sum0, sum1 = sum1, sum1 + n

        if i > 0 and sum0 <= nelec and nelec <= sum1:
            break

    if abs(sum0 - nelec) < abs(sum1 - nelec):
        homo = i - 1
        error = nelec - sum0
    else:
        homo = i
        error = nelec - sum1

    lumo = homo + 1

    if lumo == len(w):
        chempot = w[homo] + 1e-6
    else:
        chempot = 0.5 * (w[homo] + w[lumo])

    return chempot, error


def search_chempot(w, v, nphys, nelec, occupancy=2):
    """
    Search for a chemical potential, first trying with k-point
    restraints and if that doesn't succeed then without.
    """

    try:
        chempot, error = search_chempot_constrained(w, v, nphys, nelec, occupancy=occupancy)
    except ChemicalPotentialError:
        chempot, error = search_chempot_unconstrained(w, v, nphys, nelec, occupancy=occupancy)

    return chempot, error


@logging.with_timer("Chemical potential optimisation")
@logging.with_status("Optimising chemical potential")
def minimize_chempot(se, fock, nelec, occupancy=2, x0=0.0, tol=1e-6, maxiter=200):
    """
    Optimise the shift in auxiliary energies to satisfy the electron
    number, ensuring that the same shift is applied at all k-points.
    """

    tol = tol**2  # we minimize the squared error
    dtype = np.result_type(*[s.dtype for s in se], *[f.dtype for f in fock])
    nphys = max([s.nphys for s in se])
    naux = max([s.naux for s in se])
    buf = np.zeros(((nphys + naux) ** 2,), dtype=dtype)
    fargs = (se, fock, nelec, occupancy, buf)

    options = dict(maxiter=maxiter, ftol=tol, xtol=tol, gtol=tol)
    kwargs = dict(x0=x0, method="TNC", jac=True, options=options)
    fun = _gradient

    opt = scipy.optimize.minimize(fun, args=fargs, **kwargs)

    for s in se:
        s.energies -= opt.x

    ws, vs = zip(*[s.diagonalise_matrix(f) for s, f in zip(se, fock)])
    chempot = search_chempot(ws, vs, se[0].nphys, nelec, occupancy=occupancy)[0]

    for s in se:
        s.chempot = chempot

    return se, opt


class FockLoop(FockLoop):
    """
    Self-consistent loop for the density matrix via the Hartree--Fock
    self-consistent field for spin-restricted periodic systems.

    Parameters
    ----------
    gw : BaseKGW
        GW object.
    gf : tuple of dyson.Lehmann, optional
        Initial Green's function object at each k-point. If `None`, use
        `gw.init_gf()`. Default value is `None`.
    se : tuple of dyson.Lehmann, optional
        Initial self-energy object at each k-point. If passed, use as
        dynamic part of the self-energy. If `None`, self-energy is
        assumed to be static and fully defined by the Fock matrix.
        Default value is `None`.
    fock_diis_space : int, optional
        DIIS space size for the Fock matrix. Default value is `10`.
    fock_diis_min_space : int, optional
        Minimum DIIS space size for the Fock matrix. Default value is
        `1`.
    conv_tol_nelec : float, optional
        Convergence tolerance for the number of electrons. Default
        value is `1e-6`.
    conv_tol_rdm1 : float, optional
        Convergence tolerance for the density matrix. Default value is
        `1e-8`.
    max_cycle_inner : int, optional
        Maximum number of inner iterations. Default value is `100`.
    max_cycle_outer : int, optional
        Maximum number of outer iterations. Default value is `20`.
    """

    def auxiliary_shift(self, fock, se=None):
        """
        Optimise a shift in the auxiliary energies to best satisfy the
        electron number.

        Parameters
        ----------
        fock : numpy.ndarray
            Fock matrix.
        se : tuple of dyson.Lehmann, optional
            Self-energy at each k-point. If `None`, use `self.se`.
            Default value is `None`.

        Returns
        -------
        se : tuple of dyson.Lehmann
            Self-energy at each k-point.

        Notes
        -----
        If there is no dynamic part of the self-energy (`self.se` is
        `None`), this method returns `None`.
        """

        if se is None:
            se = self.se
        if se is None:
            return None

        se, opt = minimize_chempot(
            se,
            fock,
            sum(self.nelec),
            x0=se[0].chempot,
            tol=self.conv_tol_nelec,
            maxiter=self.max_cycle_inner,
        )

        return se

    def search_chempot(self, gf=None):
        """Search for a chemical potential for a given Green's function.

        Parameters
        ----------
        gf : tuple of dyson.Lehmann, optional
            Green's function at each k-point. If `None`, use `self.gf`.
            Default value is `None`.

        Returns
        -------
        chempot : float
            Chemical potential.
        nerr : float
            Error in the number of electrons.
        """

        if gf is None:
            gf = self.gf

        chempot, nerr = search_chempot(
            [g.energies for g in gf],
            [g.couplings for g in gf],
            self.nmo,
            sum(self.nelec),
        )
        nerr = abs(nerr)

        return chempot, nerr

    def solve_dyson(self, fock, se=None):
        """Solve the Dyson equation for a given Fock matrix.

        Parameters
        ----------
        fock : numpy.ndarray
            Fock matrix at each k-point.
        se : tuple of dyson.Lehmann, optional
            Self-energy at each k-point. If `None`, use `self.se`.
            Default value is `None`.

        Returns
        -------
        gf : tuple of dyson.Lehmann
            Green's function at each k-point.
        nerr : float
            Error in the number of electrons.

        Notes
        -----
        If there is no dynamic part of the self-energy (`self.se` is
        `None`), this method simply diagonalises the Fock matrix and
        returns the Lehmann representation of the resulting zeroth-order
        Green's function.
        """

        if se is None:
            se = self.se

        if se is None:
            e, c = np.linalg.eigh(fock)
        else:
            e, c = zip(*[s.diagonalise_matrix(f, chempot=0.0) for s, f in zip(se, fock)])

        e = [mpi_helper.bcast(ek, root=0) for ek in e]
        c = [mpi_helper.bcast(ck, root=0) for ck in c]

        gf = [Lehmann(ek, ck[: self.nmo], chempot=0.0) for ek, ck in zip(e, c)]

        chempot, nerr = self.search_chempot(gf)
        for k in self.kpts.loop(1):
            gf[k].chempot = chempot

        return tuple(gf), nerr

    def _density_error(self, rdm1, rdm1_prev):
        """Calculate the density error."""
        return np.max(np.absolute(rdm1 - rdm1_prev)).real

    @property
    def naux(self):
        """Get the number of auxiliary states."""
        return tuple(s.naux for s in self.se)

    @property
    def nqmo(self):
        """Get the number of quasiparticle MOs."""
        return tuple(s.nphys + s.naux for s in self.se)

    @property
    def kpts(self):
        """Get the k-points object."""
        return self.gw.kpts
