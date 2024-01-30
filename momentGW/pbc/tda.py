"""
Construct TDA moments with periodic boundary conditions.
"""

import numpy as np
import scipy.special
from pyscf import lib

from momentGW import mpi_helper, util
from momentGW.tda import dTDA as MoldTDA


class dTDA(MoldTDA):
    """
    Compute the self-energy moments using dTDA and numerical integration
    with periodic boundary conditions.

    Parameters
    ----------
    gw : BaseKGW
        GW object.
    nmom_max : int
        Maximum moment number to calculate.
    Lpx : numpy.ndarray
        Density-fitted ERI tensor, where the first two indices
        enumerate the k-points, the third index is the auxiliary
        basis function index, and the fourth and fifth indices are
        the MO and Green's function orbital indices, respectively.
    integrals : KIntegrals
        Density-fitted integrals.
    mo_energy : dict, optional
        Molecular orbital energies for each k-point. Keys are "g" and
        "w" for the Green's function and screened Coulomb interaction,
        respectively. If `None`, use `gw.mo_energy` for both. Default
        value is `None`.
    mo_occ : dict, optional
        Molecular orbital occupancies for each k-point. Keys are "g"
        and "w" for the Green's function and screened Coulomb
        interaction, respectively. If `None`, use `gw.mo_occ` for both.
        Default value is `None`.
    """

    def build_dd_moments(self):
        """Build the moments of the density-density response.

        Returns
        -------
        moments : numpy.ndarray
            Moments of the density-density response for each k-point.
        """

        cput0 = (lib.logger.process_clock(), lib.logger.perf_counter())
        lib.logger.info(self.gw, "Building density-density moments")
        lib.logger.debug(self.gw, "Memory usage: %.2f GB", self._memory_usage())

        kpts = self.kpts
        moments = np.zeros((self.nkpts, self.nkpts, self.nmom_max + 1), dtype=object)

        # Get the zeroth order moment
        for q in kpts.loop(1):
            for kj in kpts.loop(1, mpi=True):
                kb = kpts.member(kpts.wrap_around(kpts[q] + kpts[kj]))
                moments[q, kb, 0] += self.integrals.Lia[kj, kb] / self.nkpts
        cput1 = lib.logger.timer(self.gw, "zeroth moment", *cput0)

        # Get the higher order moments
        for i in range(1, self.nmom_max + 1):
            for q in kpts.loop(1):
                for kj in kpts.loop(1, mpi=True):
                    kb = kpts.member(kpts.wrap_around(kpts[q] + kpts[kj]))

                    d = util.build_1h1p_energies(
                        (self.mo_energy_w[kj], self.mo_energy_w[kb]),
                        (self.mo_occ_w[kj], self.mo_occ_w[kb]),
                    )
                    moments[q, kb, i] += moments[q, kb, i - 1] * d.ravel()[None]

                tmp = np.zeros((self.naux[q], self.naux[q]), dtype=complex)
                for ki in kpts.loop(1, mpi=True):
                    ka = kpts.member(kpts.wrap_around(kpts[q] + kpts[ki]))

                    tmp += np.dot(moments[q, ka, i - 1], self.integrals.Lia[ki, ka].T.conj())

                tmp = mpi_helper.allreduce(tmp)
                tmp *= 2.0
                tmp /= self.nkpts

                for kj in kpts.loop(1, mpi=True):
                    kb = kpts.member(kpts.wrap_around(kpts[q] + kpts[kj]))

                    moments[q, kb, i] += np.dot(tmp, self.integrals.Lai[kj, kb].conj())

            cput1 = lib.logger.timer(self.gw, "moment %d" % i, *cput1)

        return moments

    build_dd_moments_exact = build_dd_moments

    def convolve(self, eta, mo_energy_g=None, mo_occ_g=None):
        """
        Handle the convolution of the moments of the Green's function
        and screened Coulomb interaction.

        Parameters
        ----------
        eta : numpy.ndarray
            Moments of the density-density response partly transformed
            into moments of the screened Coulomb interaction for each
            k-point.
        mo_energy_g : numpy.ndarray, optional
            Energies of the Green's function for each k-point. If
            `None`, use `self.mo_energy_g`. Default value is `None`.
        mo_occ_g : numpy.ndarray, optional
            Occupancies of the Green's function for each k-point. If
            `None`, use `self.mo_occ_g`. Default value is `None`.

        Returns
        -------
        moments_occ : numpy.ndarray
            Moments of the occupied self-energy for each k-point.
        moments_vir : numpy.ndarray
            Moments of the virtual self-energy for each k-point.
        """

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

        moments_occ = np.zeros((self.nkpts, self.nmom_max + 1, nmo, nmo), dtype=complex)
        moments_vir = np.zeros((self.nkpts, self.nmom_max + 1, nmo, nmo), dtype=complex)
        moms = np.arange(self.nmom_max + 1)
        for n in moms:
            fp = scipy.special.binom(n, moms)
            fh = fp * (-1) ** moms
            for q in kpts.loop(1):
                for kp in kpts.loop(1, mpi=True):
                    kx = kpts.member(kpts.wrap_around(kpts[kp] - kpts[q]))

                    eo = np.power.outer(mo_energy_g[kx][mo_occ_g[kx] > 0], n - moms)
                    tmp_to = np.multiply(eo, fh)
                    to = lib.einsum("PQ,PQrs->rs", tmp_to, eta[kp, q][mo_occ_g[kx] > 0])
                    moments_occ[kp, n] += fproc(to)

                    ev = np.power.outer(mo_energy_g[kx][mo_occ_g[kx] == 0], n - moms)
                    tmp_tv = np.multiply(ev, fp)
                    tv = lib.einsum("PQ,PQrs->rs",tmp_tv, eta[kp, q][mo_occ_g[kx] == 0])
                    moments_vir[kp, n] += fproc(tv)

        # Numerical integration can lead to small non-hermiticity
        for n in range(self.nmom_max + 1):
            for k in kpts.loop(1, mpi=True):
                moments_occ[k, n] = 0.5 * (moments_occ[k, n] + moments_occ[k, n].T.conj())
                moments_vir[k, n] = 0.5 * (moments_vir[k, n] + moments_vir[k, n].T.conj())

        moments_occ = mpi_helper.allreduce(moments_occ)
        moments_vir = mpi_helper.allreduce(moments_vir)

        return moments_occ, moments_vir

    def build_se_moments(self, moments_dd):
        """Build the moments of the self-energy via convolution.

        Parameters
        ----------
        moments_dd : numpy.ndarray
            Moments of the density-density response for each k-point.

        Returns
        -------
        moments_occ : numpy.ndarray
            Moments of the occupied self-energy for each k-point.
        moments_vir : numpy.ndarray
            Moments of the virtual self-energy for each k-point.
        """

        cput0 = (lib.logger.process_clock(), lib.logger.perf_counter())
        lib.logger.info(self.gw, "Building self-energy moments")
        lib.logger.debug(self.gw, "Memory usage: %.2f GB", self._memory_usage())

        kpts = self.kpts

        # Setup dependent on diagonal SE
        if self.gw.diagonal_se:
            pqchar = pchar = qchar = "p"
            eta_shape = lambda k: (self.mo_energy_g[k].size, self.nmom_max + 1, self.nmo)
        else:
            pqchar, pchar, qchar = "pq", "p", "q"
            eta_shape = lambda k: (self.mo_energy_g[k].size, self.nmom_max + 1, self.nmo, self.nmo)
        eta = np.zeros((self.nkpts, self.nkpts), dtype=object)

        # Get the moments in (aux|aux) and rotate to (mo|mo)
        for n in range(self.nmom_max + 1):
            for q in kpts.loop(1):
                eta_aux = 0
                for kj in kpts.loop(1, mpi=True):
                    kb = kpts.member(kpts.wrap_around(kpts[q] + kpts[kj]))
                    eta_aux += np.dot(moments_dd[q, kb, n], self.integrals.Lia[kj, kb].T.conj())

                eta_aux = mpi_helper.allreduce(eta_aux)
                eta_aux *= 2.0
                eta_aux /= self.nkpts

                for kp in kpts.loop(1, mpi=True):
                    kx = kpts.member(kpts.wrap_around(kpts[kp] - kpts[q]))

                    if not isinstance(eta[kp, q], np.ndarray):
                        eta[kp, q] = np.zeros(eta_shape(kx), dtype=eta_aux.dtype)

                    for x in range(self.mo_energy_g[kx].size):
                        Lp = self.integrals.Lpx[kp, kx][:, :, x]
                        subscript = f"P{pchar},P{qchar}->{pqchar}"
                        tmp = np.matmul(eta_aux, Lp.conj())
                        eta[kp, q][x, n] += lib.einsum(subscript, Lp, tmp)

        cput1 = lib.logger.timer(self.gw, "rotating DD moments", *cput0)

        # Construct the self-energy moments
        moments_occ, moments_vir = self.convolve(eta)
        cput1 = lib.logger.timer(self.gw, "constructing SE moments", *cput1)

        return moments_occ, moments_vir

    build_dd_moments_exact = build_dd_moments

    @property
    def naux(self):
        """Number of auxiliaries."""
        return self.integrals.naux

    @property
    def nov(self):
        """Number of ov states in W."""
        return np.multiply.outer(
            [np.sum(occ > 0) for occ in self.mo_occ_w],
            [np.sum(occ == 0) for occ in self.mo_occ_w],
        )

    @property
    def kpts(self):
        """k-points."""
        return self.gw.kpts

    @property
    def nkpts(self):
        """Number of k-points."""
        return self.gw.nkpts
