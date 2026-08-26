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
    def build_zeroth_dd_moment(self, q, Lia=None):
        """Build the zeroth moment of the density-density response for a given set of k-points.

        Returns
        -------
        zeroth moment : numpy.ndarray
            Zeroth moment of the density-density response.
        """

        if Lia is None:
            Lia = self.integrals.Lia

        zeroth_moment = np.zeros((self.nkpts, self.nkpts), dtype=object)
        for kj in self.kpts.loop(1, mpi=True):
            kb = self.kpts.member(self.kpts.wrap_around(self.kpts[q] + self.kpts[kj]))
            zeroth_moment[q, kb] += Lia[kj, kb] / self.nkpts

        return zeroth_moment

    @logging.with_timer("Nth density-density moments")
    @logging.with_status("Constructing nth density-density moment")
    def build_nth_dd_moment(self, n, q, recursion_term=None, zeroth_mom=None, Lia=None):
        """Build the nth moment of the density-density response for a given set of k-points.

        Parameters
        ----------
        n : int
            Moment order to be built.
        q : int
            Index associated with a difference in k-points.
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

        if Lia is None:
            Lia = self.integrals.Lia

        eta_aux = 0
        if n == 0:
            if recursion_term[q, 0] != 0:
                raise AttributeError("Zeroth moment should not have a recursion term")
            if zeroth_mom is None:
                raise AttributeError(
                    "0th moment must be provided by build_zeroth_dd_moment for pbc calculations."
                )
            else:
                for kj in self.kpts.loop(1, mpi=True):
                    kb = self.kpts.member(self.kpts.wrap_around(self.kpts[q] + self.kpts[kj]))
                    recursion_term[q, kb] = zeroth_mom[q, kb]
                    eta_aux += np.dot(recursion_term[q, kb], Lia[kj, kb].T.conj())
        else:
            if recursion_term is None:
                raise AttributeError(
                    f"To build the {n}th dd-moment, a recursion_term must be provided"
                )
            kpts = self.kpts
            naux = Lia[0, kpts.member(kpts.wrap_around(kpts[q] + kpts[0]))].shape[0]
            tmp = np.zeros((naux, naux), dtype=complex)
            for ki in kpts.loop(1, mpi=True):
                ka = kpts.member(kpts.wrap_around(kpts[q] + kpts[ki]))

                tmp += np.dot(recursion_term[q, ka], Lia[ki, ka].T.conj())

            tmp = mpi_helper.allreduce(tmp)
            tmp *= 2.0 / self.nkpts
            for kj in kpts.loop(1, mpi=True):
                kb = kpts.member(kpts.wrap_around(kpts[q] + kpts[kj]))
                recursion_term[q, kb] = recursion_term[q, kb] * self.d[q, kb].ravel()[None]
                recursion_term[q, kb] += np.dot(tmp, Lia[kj, kb])  # .conj()

                eta_aux += np.dot(recursion_term[q, kb], Lia[kj, kb].T.conj())
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

    def convolve_inline(
        self,
        eta,
        kp,
        kx,
        eta_orders=None,
        mo_energy_g=None,
        mo_occ_g=None,
        moments_occ=None,
        moments_vir=None,
    ):
        """Handle the convolution of the moments of the Green's function and screened Coulomb
        interaction for a single k-point.

        Parameters
        ----------
        eta : numpy.ndarray
            Moments of the density-density response partly transformed
            into moments of the screened Coulomb interaction at each
            k-point.
        kp, kx : int
            Two indices associated with a difference in k-points.
        mo_energy_g : numpy.ndarray, optional
            Energies of the Green's function at each k-point. If
            `None`, use `self.mo_energy_g`. Default value is `None`.
        mo_occ_g : numpy.ndarray, optional
            Occupancies of the Green's function at each k-point. If
            `None`, use `self.mo_occ_g`. Default value is `None`.
        moments_occ : numpy.ndarray, optional
            Moments of the occupied self-energy at each k-point.
        moments_vir : numpy.ndarray, optional
            Moments of the virtual self-energy at each k-point.

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
        if moments_occ is None:
            moments_occ = np.zeros((self.nkpts, self.nmom_max + 1, nmo, nmo), dtype=complex)
        if moments_vir is None:
            moments_vir = np.zeros((self.nkpts, self.nmom_max + 1, nmo, nmo), dtype=complex)

        if eta_orders is None:
            eta_orders = np.arange(self.nmom_max + 1)
        eta_orders = np.asarray(eta_orders)

        for n in range(np.min(eta_orders), self.nmom_max + 1):
            # Get the binomial coefficients
            fp = scipy.special.binom(n, eta_orders)
            fh = fp * (-1) ** eta_orders
            subscript = f"t,kt,kt{pqchar}->{pqchar}"

            # Construct the occupied moments for this order
            eo = np.power.outer(mo_energy_g[kx][mo_occ_g[kx] > 0], n - eta_orders)
            to = util.einsum(subscript, fh, eo, eta[mo_occ_g[kx] > 0])
            moments_occ[kp, n] += fproc(to)

            # Construct the virtual moments for this order
            ev = np.power.outer(mo_energy_g[kx][mo_occ_g[kx] == 0], n - eta_orders)
            tv = util.einsum(subscript, fp, ev, eta[mo_occ_g[kx] == 0])
            moments_vir[kp, n] += fproc(tv)
        return moments_occ, moments_vir

    @logging.with_timer("Self-energy moments")
    @logging.with_status("Constructing self-energy moments")
    def build_se_moments(self, moments_dd=None, Lia=None):
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
        integrals = self.integrals

        # Setup dependent on diagonal SE
        if self.gw.diagonal_se:
            pqchar = pchar = qchar = "p"
            eta_shape = lambda k: (self.mo_energy_g[k].size, self.nmom_max + 1, self.nmo)
        else:
            pqchar, pchar, qchar = "pq", "p", "q"
            eta_shape = lambda k: (self.mo_energy_g[k].size, self.nmom_max + 1, self.nmo, self.nmo)
        subscript = f"P{pchar},Q{qchar},PQ->{pqchar}"

        if self.d is None:
            self._build_d()

        if self.fsc is not None:
            eta_aux_nB = None

        # Initialise the moments
        moments_occ = np.zeros((self.nkpts, self.nmom_max + 1, self.nmo, self.nmo), dtype=complex)
        moments_vir = np.zeros((self.nkpts, self.nmom_max + 1, self.nmo, self.nmo), dtype=complex)

        # Get the moments in (aux|aux) and rotate to (mo|mo)
        for q in kpts.loop(1):
            if "Lia" not in integrals._blocks or not integrals._blocks["Lia"]["built_full"]:
                integrals.get_Lia_q(q)
            if "Lpx" not in integrals._blocks or not integrals._blocks["Lpx"]["built_full"]:
                integrals.get_Lpx_q(q)
            for n in range(self.nmom_max + 1):
                if moments_dd is None:
                    if n == 0:
                        zeroth_mom = self.build_zeroth_dd_moment(q)
                        recursion_term = np.zeros_like(zeroth_mom)
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

                if self.fsc is not None and q == 0 and "B" not in self.fsc:
                    if n == 0:
                        zeroth_mom_nB = self.build_zeroth_dd_moment(q, Lia=self.integrals.Mia)
                        recursion_term_nB = np.zeros_like(zeroth_mom_nB)
                    recursion_term_nB, eta_aux_nB = self.build_nth_dd_moment(
                        n, q, recursion_term_nB, zeroth_mom_nB, Lia=self.integrals.Mia
                    )

                for kp in kpts.loop(1, mpi=True):
                    kx = kpts.member(kpts.wrap_around(kpts[kp] - kpts[q]))

                    # if not isinstance(eta[kp, q], np.ndarray):
                    eta = np.zeros(eta_shape(kx), dtype=eta_aux.dtype)

                    for x in range(self.mo_energy_g[kx].size):
                        Lp = self.integrals.Lpx[kp, kx][:, :, x]
                        if q == 0 and self.fsc is not None:
                            wing_tmp = util.einsum(
                                f"P,P{pchar}{qchar}->{pqchar}",
                                eta_aux[0, 1:],
                                self.integrals.Lpx[kp, kx],
                            )
                            eta = self.get_fsc_terms(
                                eta_aux, eta, Lp, x, n, subscript, wing_tmp, eta_aux_nB
                            )
                        else:
                            eta[x, n] += util.einsum(subscript, Lp, Lp.conj(), eta_aux)

                    moments_occ, moments_vir = self.convolve_inline(
                        eta,
                        kp,
                        kx,
                        eta_orders=[n],
                        moments_occ=moments_occ,
                        moments_vir=moments_vir,
                    )

        # Construct the self-energy moments
        moments_occ, moments_vir = self.hermiticity_correction(moments_occ, moments_vir)

        # Sum over all processes
        moments_occ = mpi_helper.allreduce(moments_occ)
        moments_vir = mpi_helper.allreduce(moments_vir)

        # moments_occ, moments_vir = self.convolve(eta)

        return moments_occ, moments_vir

    def hermiticity_correction(self, moments_occ, moments_vir):
        """Correction for small errors in the hermiticty of the moments due to the numerical
        integration.

        Parameters
        ----------
        moments_occ : numpy.ndarray
            Moments of the occupied self-energy at each k-point.
        moments_vir : numpy.ndarray
            Moments of the virtual self-energy at each k-point.

        Returns
        -------
        moments_occ : numpy.ndarray
            Moments of the occupied self-energy at each k-point.
        moments_vir : numpy.ndarray
            Moments of the virtual self-energy at each k-point.
        """
        # Numerical integration can lead to small non-hermiticity
        for n in range(self.nmom_max + 1):
            for k in self.kpts.loop(1):
                moments_occ[k, n] = 0.5 * (moments_occ[k, n] + moments_occ[k, n].T.conj())
                moments_vir[k, n] = 0.5 * (moments_vir[k, n] + moments_vir[k, n].T.conj())
        return moments_occ, moments_vir

    def get_fsc_terms(self, eta_aux, eta, Lp, x, n, subscript, wing_tmp, eta_aux_nB=None):
        """Construct the Head, Wings and Body corrections.

        Parameters
        ----------
        eta : numpy.ndarray
            Moments of the density-density response at a k-point difference.
        eta : numpy.ndarray
            Moments of the density-density response partly transformed
            into moments of the screened Coulomb interaction at each
            k-point.
        Lp : numpy.ndarray
            A single k-point ``(aux, MO)`` array at a chosen ``x``.
        x : int
            Index of ``(MO)`` .
        n : int
            Index of current moment.
        subscript : str
            Einsum shape depending on whether it is a diagonal calculation.
        wing_tmp : numpy.ndarray
            Calculated wing term.
        eta_aux_nB: numpy.ndarray
            Moments of the density-density response at a k-point difference without the plane wave
            approximation.

        Returns
        -------
        eta : numpy.ndarray
            Moments of the density-density response partly transformed
            into moments of the screened Coulomb interaction at each
            k-point.
        """
        q0 = (6 * np.pi**2 / (self.kpts.cell.vol * self.nkpts)) ** (1 / 3)

        if "B" not in self.fsc:
            eta[x, n] += util.einsum(subscript, Lp, Lp.conj(), eta_aux_nB)
        else:
            eta[x, n] += util.einsum(subscript, Lp, Lp.conj(), eta_aux[1:, 1:])

        if "H" in self.fsc:
            if self.gw.diagonal_se:
                eta[x, n][x] += (2 / np.pi) * q0 * eta_aux[0, 0] * self.nkpts
            else:
                eta[x, n][x, x] += (2 / np.pi) * q0 * eta_aux[0, 0] * self.nkpts

        if "W" in self.fsc:
            wing_tmp = (
                ((q0**2) * ((self.kpts.cell.vol / (4 * np.pi**3)) ** (1 / 2)))
                * wing_tmp.real
                * self.nkpts
            )
            if self.gw.diagonal_se:
                eta[x, n][x] -= 2 * wing_tmp[x]
            else:
                eta[x, n][x, :] -= wing_tmp.T[x, :]
                eta[x, n][:, x] -= wing_tmp[:, x]
        return eta

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
