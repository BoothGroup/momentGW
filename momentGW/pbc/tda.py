"""
Construct TDA moments with periodic boundary conditions.
"""

import numpy as np
import scipy.special

from momentGW import logging, mpi_helper, util
from momentGW.tda import dTDA as MoldTDA


class dTDA(MoldTDA):
    """
    Compute the self-energy moments using dTDA with periodic boundary
    conditions.

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

    def __init__(
        self,
        gw,
        nmom_max,
        integrals,
        mo_energy=None,
        mo_occ=None,
        fsc=False,
    ):
        super().__init__(gw, nmom_max, integrals, mo_energy=mo_energy, mo_occ=mo_occ)
        self.fsc = fsc

    @logging.with_timer("Density-density moments")
    @logging.with_status("Constructing density-density moments")
    def build_dd_moments(self):
        """Build the moments of the density-density response.

        Returns
        -------
        moments : numpy.ndarray
            Moments of the density-density response at each k-point.
        """

        moments = self.build_uncorrected_dd_moments()

        if self.fsc is not None:
            corrected_moments = self.build_corrected_dd_moments()
            for i in range(self.nmom_max + 1):
                for kj in self.kpts.loop(1, mpi=True):
                    if "B" in self.fsc:
                        moments[0, kj, i] = corrected_moments[kj, i]
                    else:
                        moments[0, kj, i] = np.concatenate(
                            (np.array([corrected_moments[kj, i][0, :]]), moments[0, kj, i])
                        )

            return moments
        else:
            return moments

    def build_uncorrected_dd_moments(self):
        """Build the moments of the density-density response for
        all k-point pairs.

        Returns
        -------
        moments : numpy.ndarray
            Moments of the density-density response at each k-point pair.
        """

        # Initialise the moments
        kpts = self.kpts
        moments = np.zeros((self.nkpts, self.nkpts, self.nmom_max + 1), dtype=object)

        # Get the zeroth order moment
        for q in kpts.loop(1):
            for kj in kpts.loop(1, mpi=True):
                kb = kpts.member(kpts.wrap_around(kpts[q] + kpts[kj]))
                moments[q, kb, 0] += self.integrals.Lia[kj, kb] / self.nkpts

        # Get the higher order moments
        for i in range(1, self.nmom_max + 1):
            for q in kpts.loop(1):
                tmp = np.zeros((self.naux[q], self.naux[q]), dtype=complex)
                for kj in kpts.loop(1, mpi=True):
                    kb = kpts.member(kpts.wrap_around(kpts[q] + kpts[kj]))

                    d = util.build_1h1p_energies(
                        (self.mo_energy_w[kj], self.mo_energy_w[kb]),
                        (self.mo_occ_w[kj], self.mo_occ_w[kb]),
                    )
                    moments[q, kb, i] += moments[q, kb, i - 1] * d.ravel()[None]

                    tmp += np.dot(moments[q, kb, i - 1], self.integrals.Lia[kj, kb].T.conj())

                tmp = mpi_helper.allreduce(tmp)
                tmp *= 2.0
                tmp /= self.nkpts

                for kj in kpts.loop(1, mpi=True):
                    kb = kpts.member(kpts.wrap_around(kpts[q] + kpts[kj]))

                    moments[q, kb, i] += np.dot(tmp, self.integrals.Lai[kj, kb].conj())

        return moments

    def build_corrected_dd_moments(self):
        """Build the finite size corrections for the moments of the
        density-density response.

        Returns
        -------
        moments : numpy.ndarray
            Corrected moments of the density-density response at each
            k-point for q=0.
        """
        kpts = self.kpts
        moments = np.zeros((self.nkpts, self.nmom_max + 1), dtype=object)

        # Get the zeroth order moment
        for kj in kpts.loop(1, mpi=True):
            moments[kj, 0] += self.integrals.Mia[kj] / self.nkpts

        # Get the higher order moments
        for i in range(1, self.nmom_max + 1):
            tmp = np.zeros((self.naux[0] + 1, self.naux[0] + 1), dtype=complex)
            for kj in kpts.loop(1, mpi=True):
                d = util.build_1h1p_energies(
                    (self.mo_energy_w[kj], self.mo_energy_w[kj]),
                    (self.mo_occ_w[kj], self.mo_occ_w[kj]),
                )
                moments[kj, i] += moments[kj, i - 1] * d.ravel()
                tmp += util.einsum("Pa,aQ->PQ", moments[kj, i - 1], self.integrals.Mia[kj].T.conj())
            tmp = mpi_helper.allreduce(tmp)
            tmp *= 2.0
            tmp /= self.nkpts

            for kj in kpts.loop(1, mpi=True):
                moments[kj, i] += util.einsum("PQ,Qa->Pa", tmp, self.integrals.Mia[kj])

        return moments

    def kernel(self, exact=False):
        """
        Run the polarizability calculation to compute moments of the
        self-energy.

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
        """
        Handle the convolution of the moments of the Green's function
        and screened Coulomb interaction.

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
    def build_se_moments(self, moments_dd):
        """Build the moments of the self-energy via convolution.

        Parameters
        ----------
        moments_dd : numpy.ndarray
            Moments of the density-density response at each k-point.

        Returns
        -------
        moments_occ : numpy.ndarray
            Moments of the occupied self-energy at each k-point.
        moments_vir : numpy.ndarray
            Moments of the virtual self-energy at each k-point.
        """
        kpts = self.kpts

        if self.fsc is not None:
            q0 = (6 * np.pi**2 / (kpts.cell.vol * self.nkpts)) ** (1 / 3)

        # Setup dependent on diagonal SE
        if self.gw.diagonal_se:
            pqchar = pchar = qchar = "p"
            eta_shape = lambda k: (self.mo_energy_g[k].size, self.nmom_max + 1, self.nmo)
        else:
            pqchar, pchar, qchar = "pq", "p", "q"
            eta_shape = lambda k: (self.mo_energy_g[k].size, self.nmom_max + 1, self.nmo, self.nmo)
        eta = np.zeros((self.nkpts, self.nkpts), dtype=object)

        for n in range(self.nmom_max + 1):
            for q in kpts.loop(1):
                eta_aux = 0
                for kj in kpts.loop(1, mpi=True):
                    if q == 0 and self.fsc is not None:
                        eta_aux += np.dot(moments_dd[q, kj, n], self.integrals.Mia[kj].T.conj())
                    else:
                        kb = kpts.member(kpts.wrap_around(kpts[q] + kpts[kj]))
                        eta_aux += np.dot(moments_dd[q, kb, n], self.integrals.Lia[kj, kb].T.conj())

                eta_aux = mpi_helper.allreduce(eta_aux)
                eta_aux *= 2.0

                for kp in kpts.loop(1, mpi=True):
                    kx = kpts.member(kpts.wrap_around(kpts[kp] - kpts[q]))

                    if not isinstance(eta[kp, q], np.ndarray):
                        eta[kp, q] = np.zeros(eta_shape(kx), dtype=eta_aux.dtype)

                    for x in range(self.mo_energy_g[kx].size):
                        Lp = self.integrals.Lpx[kp, kx][:, :, x]
                        subscript = f"P{pchar},Q{qchar},PQ->{pqchar}"
                        if q == 0 and self.fsc is not None:
                            eta[kp, q][x, n] += util.einsum(
                                subscript, Lp, Lp.conj(), eta_aux[1:, 1:] / self.nkpts
                            )
                            if "H" in self.fsc and mpi_helper.rank == 0:
                                if self.gw.diagonal_se:
                                    eta[kp, q][x, n][x] += (2 / np.pi) * q0 * eta_aux[0, 0]
                                else:
                                    eta[kp, q][x, n][x, x] += (2 / np.pi) * q0 * eta_aux[0, 0]
                            if "W" in self.fsc:
                                wing_tmp = util.einsum(
                                    f"P,P{pchar}{qchar}->{pqchar}",
                                    eta_aux[0, 1:],
                                    self.integrals.Lpx[kp, kx],
                                )
                                wing_tmp = (
                                    (q0**2) * ((kpts.cell.vol / (4 * np.pi**3)) ** (1 / 2))
                                ) * wing_tmp.real
                                if self.gw.diagonal_se:
                                    eta[kp, q][x, n][x] += 2 * wing_tmp[x]
                                else:
                                    eta[kp, q][x, n][x, :] -= wing_tmp.T[x, :]
                                    eta[kp, q][x, n][:, x] -= wing_tmp[:, x]
                        else:
                            eta[kp, q][x, n] += util.einsum(
                                subscript, Lp, Lp.conj(), eta_aux / self.nkpts
                            )

        # Construct the self-energy moments
        moments_occ, moments_vir = self.convolve(eta)

        return moments_occ, moments_vir

    @property
    def naux(self):
        """Number of auxiliaries."""
        return self.integrals.naux

    @property
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

    @property
    def q_abs(self):
        """Get the absolute value of a region near the origin (q=0)."""
        return self.kpts.cell.get_abs_kpts(np.array([1e-3, 0, 0]).reshape(1, 3))
