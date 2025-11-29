"""Construct TDA moments with periodic boundary conditions."""

import functools

import numpy as np
import scipy.special

from momentGW import logging, mpi_helper, util
from momentGW.tda import dTDA as MoldTDA


class dTDA(MoldTDA):
    """Compute the self-energy moments using dTDA with periodic boundary conditions.

    Parameters
    ----------
    gw : BaseKGW
        GW object.
    nmom_max : int
        Maximum moment number to calculate.
    integrals : KIntegrals
        Density-fitted integrals at each k-point.
    mo_energy : dict, optional
        Molecular orbital energies at each k-point. Keys are "g" and
        "w" for the Green's function and screened Coulomb interaction,
        respectively. If `None`, use `gw.mo_energy` for both. Default
        value is `None`.
    mo_occ : dict, optional
        Molecular orbital occupancies at each k-point. Keys are "g"
        and "w" for the Green's function and screened Coulomb
        interaction, respectively. If `None`, use `gw.mo_occ` for both.
        Default value is `None`.
    """

    def _build_d(self):
        """Construct the energy differences matrix.

        Returns
        -------
        d : numpy.ndarray
            Orbital energy differences at each k-point.
        """

        self.d = np.zeros((self.nkpts, self.nkpts), dtype=object)

        for q in self.kpts.loop(1):
            for ki in self.kpts.loop(1, mpi=True):
                ka = self.kpts.member(self.kpts.wrap_around(self.kpts[q] + self.kpts[ki]))
                self.d[q, ka] = util.build_1h1p_energies(
                    (self.mo_energy_w[ki], self.mo_energy_w[ka]),
                    (self.mo_occ_w[ki], self.mo_occ_w[ka]),
                ).ravel()

    @logging.with_timer("Zeroth density-density moments")
    @logging.with_status("Constructing zeroth density-density moment")
    def build_zeroth_moment(self):
        """Build the zeroth moment of the density-density response for a given set of k-points.

        Returns
        -------
        zeroth moment : numpy.ndarray
            Zeroth moment of the density-density response.
        """
        zeroth_moment = np.zeros((self.nkpts, self.nkpts), dtype=object)
        for q in self.kpts.loop(1):
            for kj in self.kpts.loop(1, mpi=True):
                kb = self.kpts.member(self.kpts.wrap_around(self.kpts[q] + self.kpts[kj]))
                zeroth_moment[q, kb] += self.integrals.Lia[kj, kb] / self.nkpts

        return zeroth_moment

    @logging.with_timer("Nth density-density moments")
    @logging.with_status("Constructing nth density-density moment")
    def build_nth_dd_moment(self, n, q, recursion_term=None, zeroth_mom=None):
        """Build the nth moment of the density-density response for a given set of k-points.

        Parameters
        ----------
        n : int
            Moment order to be built.
        q : int
            Index associated with a difference in k-points
        recursion_term : numpy.ndarray, optional
            Previous recursion term required to build the next moment. In the case of TDA this is
            the previous density-density response.
        zeroth moment : numpy.ndarray, optional
            Zeroth moment of the density-density response.

        Returns
        -------
        recursion_term : numpy.ndarray
            Term required for the next moment. In the case of TDA this is the current
            density-density response moment.
        eta_aux : numpy.ndarray
            The nth density-density response moment in (N_aux,N_aux) form
        """
        eta_aux = 0
        if n == 0:
            if recursion_term[q, 0] != 0:
                raise AttributeError("Zeroth moment should not have a recursion term")
            if zeroth_mom is None:
                raise AttributeError(
                    "0th moment must be provided by build_zeroth_moment for k-point calculations."
                )
            else:
                for kj in self.kpts.loop(1, mpi=True):
                    kb = self.kpts.member(self.kpts.wrap_around(self.kpts[q] + self.kpts[kj]))
                    recursion_term[q, kb] = zeroth_mom[q, kb]
                    eta_aux += np.dot(recursion_term[q, kb], self.integrals.Lia[kj, kb].T.conj())
        else:
            if recursion_term is None:
                raise AttributeError(
                    f"To build the {n}th dd-moment, a recursion_term must be provided"
                )
            kpts = self.kpts
            tmp = np.zeros((self.naux[q], self.naux[q]), dtype=complex)
            for ki in kpts.loop(1, mpi=True):
                ka = kpts.member(kpts.wrap_around(kpts[q] + kpts[ki]))

                tmp += np.dot(recursion_term[q, ka], self.integrals.Lia[ki, ka].T.conj())

            tmp = mpi_helper.allreduce(tmp)
            tmp *= 2.0 / self.nkpts
            for kj in kpts.loop(1, mpi=True):
                kb = kpts.member(kpts.wrap_around(kpts[q] + kpts[kj]))
                recursion_term[q, kb] = recursion_term[q, kb] * self.d[q, kb].ravel()[None]
                recursion_term[q, kb] += np.dot(tmp, self.integrals.Lai[kj, kb].conj())

                eta_aux += np.dot(recursion_term[q, kb], self.integrals.Lia[kj, kb].T.conj())
            del tmp

        eta_aux = mpi_helper.allreduce(eta_aux)
        eta_aux *= 2.0 / self.nkpts
        return recursion_term, eta_aux

    @logging.with_timer("Density-density moments")
    @logging.with_status("Constructing density-density moments")
    def build_dd_moments(self):
        """Build the moments of the density-density response.

        Returns
        -------
        moments : numpy.ndarray
            Moments of the density-density response at each k-point.
        """

        if self.d is None:
            self._build_d()

        # Initialise the moments
        kpts = self.kpts
        naux = self.naux
        moments = np.zeros((self.nkpts, self.nkpts, self.nmom_max + 1), dtype=object)

        # Get the zeroth order moment
        for q in kpts.loop(1):
            for kj in kpts.loop(1, mpi=True):
                kb = kpts.member(kpts.wrap_around(kpts[q] + kpts[kj]))
                moments[q, kb, 0] += self.integrals.Lia[kj, kb] / self.nkpts

        # Get the higher order moments
        for i in range(1, self.nmom_max + 1):
            for q in kpts.loop(1):
                tmp = np.zeros((naux[q], naux[q]), dtype=complex)
                for ki in kpts.loop(1, mpi=True):
                    ka = kpts.member(kpts.wrap_around(kpts[q] + kpts[ki]))

                    tmp += np.dot(moments[q, ka, i - 1], self.integrals.Lia[ki, ka].T.conj())

                tmp = mpi_helper.allreduce(tmp)
                tmp *= 2.0 / self.nkpts

                for kj in kpts.loop(1, mpi=True):
                    kb = kpts.member(kpts.wrap_around(kpts[q] + kpts[kj]))
                    d = util.build_1h1p_energies(
                        (self.mo_energy_w[kj], self.mo_energy_w[kb]),
                        (self.mo_occ_w[kj], self.mo_occ_w[kb]),
                    )
                    moments[q, kb, i] += moments[q, kb, i - 1] * d.ravel()[None]

                    moments[q, kb, i] += np.dot(tmp, self.integrals.Lai[kj, kb].conj())

        return moments

    def kernel(self, exact=False):
        """Run the polarizability calculation to compute moments of the self-energy.

        Parameters
        ----------
        exact : bool, optional
            Has no effect and is only present for compatibility with
            `dRPA`. Default value is `False`.

        Returns
        -------
        moments_occ : numpy.ndarray
            Moments of the occupied self-energy at each k-point.
        moments_vir : numpy.ndarray
            Moments of the virtual self-energy at each k-point.
        """
        return super().kernel(exact=exact)

    @logging.with_timer("Moment convolution")
    @logging.with_status("Convoluting moments")
    def convolve(self, eta, mo_energy_g=None, mo_occ_g=None):
        """Handle the convolution of the moments of the Green's function and screened Coulomb
        interaction.

        Parameters
        ----------
        eta : numpy.ndarray
            Moments of the density-density response partly transformed
            into moments of the screened Coulomb interaction at each
            k-point.
        mo_energy_g : numpy.ndarray, optional
            Energies of the Green's function at each k-point. If
            `None`, use `self.mo_energy_g`. Default value is `None`.
        mo_occ_g : numpy.ndarray, optional
            Occupancies of the Green's function at each k-point. If
            `None`, use `self.mo_occ_g`. Default value is `None`.

        Returns
        -------
        moments_occ : numpy.ndarray
            Moments of the occupied self-energy at each k-point.
        moments_vir : numpy.ndarray
            Moments of the virtual self-energy at each k-point.
        """

        # Get the orbitals
        if mo_energy_g is None:
            mo_energy_g = self.mo_energy_g
        if mo_occ_g is None:
            mo_occ_g = self.mo_occ_g
        kpts = self.kpts

        # Setup dependent on diagonal SE
        if self.gw.diagonal_se:
            pqchar = "p"
            fproc = lambda x: np.diag(x)
        else:
            pqchar = "pq"
            fproc = lambda x: x

        # We avoid self.nmo for inheritence reasons, but in MPI eta is
        # sparse, hence this weird code
        for part in eta.ravel():
            if isinstance(part, np.ndarray):
                nmo = part.shape[-1]
                break

        # Initialise the moments
        moments_occ = np.zeros((self.nkpts, self.nmom_max + 1, nmo, nmo), dtype=complex)
        moments_vir = np.zeros((self.nkpts, self.nmom_max + 1, nmo, nmo), dtype=complex)

        moms = np.arange(self.nmom_max + 1)
        for n in moms:
            # Get the binomial coefficients
            fp = scipy.special.binom(n, moms)
            fh = fp * (-1) ** moms

            for q in kpts.loop(1):
                for kp in kpts.loop(1, mpi=True):
                    kx = kpts.member(kpts.wrap_around(kpts[kp] - kpts[q]))
                    subscript = f"t,kt,kt{pqchar}->{pqchar}"

                    # Construct the occupied moments for this order
                    eo = np.power.outer(mo_energy_g[kx][mo_occ_g[kx] > 0], n - moms)
                    to = util.einsum(subscript, fh, eo, eta[kp, q][mo_occ_g[kx] > 0])
                    moments_occ[kp, n] += fproc(to)

                    # Construct the virtual moments for this order
                    ev = np.power.outer(mo_energy_g[kx][mo_occ_g[kx] == 0], n - moms)
                    tv = util.einsum(subscript, fp, ev, eta[kp, q][mo_occ_g[kx] == 0])
                    moments_vir[kp, n] += fproc(tv)

        # Numerical integration can lead to small non-hermiticity
        for n in range(self.nmom_max + 1):
            for k in kpts.loop(1, mpi=True):
                moments_occ[k, n] = 0.5 * (moments_occ[k, n] + moments_occ[k, n].T.conj())
                moments_vir[k, n] = 0.5 * (moments_vir[k, n] + moments_vir[k, n].T.conj())

        # Sum over all processes
        moments_occ = mpi_helper.allreduce(moments_occ)
        moments_vir = mpi_helper.allreduce(moments_vir)

        return moments_occ, moments_vir

    @logging.with_timer("Self-energy moments")
    @logging.with_status("Constructing self-energy moments")
    def build_se_moments(self, moments_dd=None):
        """Build the moments of the self-energy via convolution.

        Parameters
        ----------
        moments_dd : numpy.ndarray, optional
            Moments of the density-density response at each k-point.

        Returns
        -------
        moments_occ : numpy.ndarray
            Moments of the occupied self-energy at each k-point.
        moments_vir : numpy.ndarray
            Moments of the virtual self-energy at each k-point.
        """

        kpts = self.kpts

        # Setup dependent on diagonal SE
        if self.gw.diagonal_se:
            pqchar = pchar = qchar = "p"
            eta_shape = lambda k: (self.mo_energy_g[k].size, self.nmom_max + 1, self.nmo)
        else:
            pqchar, pchar, qchar = "pq", "p", "q"
            eta_shape = lambda k: (self.mo_energy_g[k].size, self.nmom_max + 1, self.nmo, self.nmo)
        eta = np.zeros((self.nkpts, self.nkpts), dtype=object)

        if self.d is None:
            self._build_d()

        if moments_dd is None:
            zeroth_mom = self.build_zeroth_moment()
            recursion_term = np.zeros_like(zeroth_mom)

        # Get the moments in (aux|aux) and rotate to (mo|mo)
        for n in range(self.nmom_max + 1):
            for q in kpts.loop(1):
                if moments_dd is None:
                    recursion_term, eta_aux = self.build_nth_dd_moment(
                        n, q, recursion_term, zeroth_mom
                    )
                else:
                    eta_aux = 0
                    for kj in kpts.loop(1, mpi=True):
                        kb = kpts.member(kpts.wrap_around(kpts[q] + kpts[kj]))
                        eta_aux += np.dot(moments_dd[q, kb, n], self.integrals.Lia[kj, kb].T.conj())

                    eta_aux = mpi_helper.allreduce(eta_aux)
                    eta_aux *= 2.0 / self.nkpts

                for kp in kpts.loop(1, mpi=True):
                    kx = kpts.member(kpts.wrap_around(kpts[kp] - kpts[q]))

                    if not isinstance(eta[kp, q], np.ndarray):
                        eta[kp, q] = np.zeros(eta_shape(kx), dtype=eta_aux.dtype)

                    for x in range(self.mo_energy_g[kx].size):
                        Lp = self.integrals.Lpx[kp, kx][:, :, x]
                        subscript = f"P{pchar},Q{qchar},PQ->{pqchar}"
                        eta[kp, q][x, n] += util.einsum(subscript, Lp, Lp.conj(), eta_aux)

        # Construct the self-energy moments
        moments_occ, moments_vir = self.convolve(eta)

        return moments_occ, moments_vir

    @functools.cached_property
    def nov(self):
        """Get the number of ov states in W."""
        return np.multiply.outer(
            [np.sum(occ > 0) for occ in self.mo_occ_w],
            [np.sum(occ == 0) for occ in self.mo_occ_w],
        )

    @property
    def kpts(self):
        """Get the k-points."""
        return self.gw.kpts

    @property
    def nkpts(self):
        """Get the number of k-points."""
        return self.gw.nkpts
