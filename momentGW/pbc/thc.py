"""
THC integral helpers with periodic boundary conditions.
"""

import h5py
import numpy as np
from pyscf import lib
from scipy.special import binom

from momentGW.pbc.ints import KIntegrals as KIntegrals_gen
from momentGW.pbc.tda import dTDA as TDA_gen
from momentGW.thc import Integrals
from momentGW.thc import dTDA as MolTDA


class KIntegrals(Integrals, KIntegrals_gen):
    """
    Container for the THC integrals required for KGW methods.
    """

    def __init__(
        self,
        with_df,
        kpts,
        mo_coeff,
        mo_occ,
        file_path=None,
    ):
        Integrals.__init__(
            self,
            with_df,
            mo_coeff,
            mo_occ,
            file_path=file_path,
        )
        self.kpts = kpts
        self.compression = None
        self._madelung = None

    def import_ints(self):
        """
        Imports a h5py file containing a dictionary. Inside the dict, a
        'collocation_matrix' and a 'coulomb_matrix' must be contained
        with shapes (MO, aux) and (aux,aux) respectively.
        """
        if self.file_path is None:
            raise ValueError("file path cannot be None for THC implementation")

        thc_eri = h5py.File(self.file_path, "r")

        kpts_imp = np.array(thc_eri["kpts"])

        if kpts_imp.shape[0] != len(self.kpts):
            raise ValueError("Number of kpts imported differs from pyscf")
        if not np.allclose(kpts_imp, self.kpts._kpts) and not np.allclose(
            kpts_imp, -self.kpts._kpts
        ):
            raise ValueError("Different kpts imported to those in pyscf")

        cou = {}
        coll = {}
        for ki in range(len(self.kpts)):
            cou[ki] = np.array(thc_eri["coulomb_matrix"])[ki, ..., 0]
            # coll[ki] = np.array(thc_eri["collocation_matrix"])[count : (count + self.nmo), ..., 0].T
            # Here since they changed file structure
            coll[ki] = np.array(thc_eri["collocation_matrix"])[0, ki, ..., 0]

        self._blocks["coll"] = coll
        self._blocks["cou"] = cou

    def transform(self, do_Lpq=True, do_Lpx=True, do_Lia=True):
        """
        Transform the integrals.

        Parameters
        ----------
        do_Lpq : bool
            If `True` contrstructs the Lp array using the mo_coeff and
            the collocation matrix. Default value is `True`. Required
            for the initial creation.
        do_Lpx : bool
            If `True` contrstructs the Lx array using the mo_coeff_g and
            the collocation matrix. Default value is `True`.
        do_Lia : bool
            If `True` contrstructs the Li and La arrays using the
            mo_coeff_w and the collocation matrix. Default value is
            `True`.
        """

        if not any([do_Lpq, do_Lpx, do_Lia]):
            return
        if self.coll is None and self.cou is None:
            self.import_ints()

        Lp = {}
        Lx = {}
        Li = {}
        La = {}

        for ki in range(self.nkpts):
            if do_Lpq:
                Lp[ki] = lib.einsum("Lp,pq->Lq", self.coll[ki], self.mo_coeff[ki])

            if do_Lpx:
                Lx[ki] = lib.einsum("Lp,pq->Lq", self.coll[ki], self.mo_coeff_g[ki])

            if do_Lia:
                ci = self.mo_coeff_w[ki][:, self.mo_occ_w[ki] > 0]
                ca = self.mo_coeff_w[ki][:, self.mo_occ_w[ki] == 0]

                Li[ki] = lib.einsum("Lp,pi->Li", self.coll[ki], ci)
                La[ki] = lib.einsum("Lp,pa->La", self.coll[ki], ca)

        if do_Lpq:
            self._blocks["Lp"] = Lp
        if do_Lpx:
            self._blocks["Lx"] = Lx
        if do_Lia:
            self._blocks["Li"] = Li
            self._blocks["La"] = La

    def get_j(self, dm, basis="mo"):
        """Build the J matrix.

        Parameters
        ----------
        dm : numpy.ndarray
            Density matrix.
        basis : str, optional
            Basis in which to build the J matrix. One of
            `("ao", "mo")`. Default value is `"mo"`.

        Returns
        -------
        vj : numpy.ndarray
            J matrix.

        Notes
        -----
        The basis of `dm` must be the same as `basis`.
        """

        assert basis in ("ao", "mo")

        vj = np.zeros_like(dm, dtype=complex)
        if basis == "ao":
            if self.coll is None and self.cou is None:
                self.import_ints()
            Lp = self.coll
            cou = self.cou
        else:
            Lp = self.Lp
            cou = self.cou

        buf = 0.0
        for ki in range(self.nkpts):
            tmp = lib.einsum("pq,Kp,Kq->K", dm[ki], Lp[ki], Lp[ki].conj())
            tmp = lib.einsum("K,KL->L", tmp, cou[0])
            buf += tmp

        buf /= self.nkpts

        for kj in range(self.nkpts):
            vj[kj] = lib.einsum("L,Lr,Ls->rs", buf, Lp[kj].conj(), Lp[kj])
        return vj

    def get_k(self, dm, basis="mo"):
        """Build the K matrix.

        Parameters
        ----------
        dm : numpy.ndarray
            Density matrix.
        basis : str, optional
            Basis in which to build the K matrix. One of
            `("ao", "mo")`. Default value is `"mo"`.

        Returns
        -------
        vk : numpy.ndarray
            K matrix.

        Notes
        -----
        The basis of `dm` must be the same as `basis`.
        """

        assert basis in ("ao", "mo")

        vk = np.zeros_like(dm, dtype=complex)
        if basis == "ao":
            if self.coll is None and self.cou is None:
                self.import_ints()
            Lp = self.coll
            cou = self.cou
        else:
            Lp = self.Lp
            cou = self.cou

        buf = np.zeros((self.nkpts, self.nkpts, self.naux, self.naux), dtype=complex)
        for ki in range(self.nkpts):
            for kk in range(self.nkpts):
                tmp = lib.einsum("pq,Kp->Kq", dm[kk], Lp[kk].conj())
                tmp = lib.einsum("Kq,Lq->KL", tmp, Lp[kk])
                kb = self.kpts.member(self.kpts.wrap_around(self.kpts[ki] + self.kpts[kk]))
                buf[ki, kk] = lib.einsum("KL,KL->KL", tmp, cou[kb])
        buf /= self.nkpts
        for ki in range(self.nkpts):
            for kk in range(self.nkpts):
                tmp = lib.einsum("KL,Ks->Ls", buf[ki, kk], Lp[ki].conj())
                vk[ki] += lib.einsum("Ls,Lr->rs", tmp, Lp[ki])
        return vk

    @property
    def nkpts(self):
        """Number of k points"""
        return len(self.kpts)

    @property
    def naux(self):
        """Return the number of auxiliary basis functions."""
        return self.cou[0].shape[0]


class dTDA(MolTDA, TDA_gen):
    """
    Compute the self-energy moments using dTDA and numerical integration
    with tensor-hypercontraction and periodic boundary conditions.

    Parameters
    ----------
    gw : BaseKGW
        GW object.
    nmom_max : int
        Maximum moment number to calculate.
    integrals : KIntegrals
        Density-fitted integrals.
    mo_energy : numpy.ndarray or tuple of numpy.ndarray, optional
        Molecular orbital energies at each k-point.  If a tuple is passed,
        the first element corresponds to the Green's function basis and
        the second to the screened Coulomb interaction.  Default value is
        that of `gw.mo_energy`.
    mo_occ : numpy.ndarray or tuple of numpy.ndarray, optional
        Molecular orbital occupancies at each k-point.  If a tuple is
        passed, the first element corresponds to the Green's function basis
        and the second to the screened Coulomb interaction.  Default value
        is that of `gw.mo_occ`.
    """

    def build_dd_moments(self):
        """
        Build the moments of the density-density response using
        tensor-hypercontraction in k-space.

        Returns
        -------
        moments : numpy.ndarray
            Moments of the density-density response.

        Notes
        -----
        Unlike the standard `momentGW.tda` implementation, this method
        scales as :math:`O(N^3)` with system size instead of
        :math:`O(N^4)`.
        """

        cput0 = (lib.logger.process_clock(), lib.logger.perf_counter())
        lib.logger.info(self.gw, "Building density-density moments")
        lib.logger.debug(self.gw, "Memory usage: %.2f GB", self._memory_usage())

        zeta = np.zeros((self.nkpts, self.nkpts, self.nmom_max + 1), dtype=object)

        kpts = self.kpts
        cou_occ = np.zeros((self.nkpts, 1), dtype=object)
        cou_vir = np.zeros((self.nkpts, 1), dtype=object)

        cou_d_left = np.zeros((self.nkpts, self.nkpts, self.nmom_max + 1), dtype=object)
        cou_d_only = np.zeros((self.nkpts, self.nkpts, self.nmom_max + 1), dtype=object)
        cou_left = np.zeros((self.nkpts, self.nkpts, 1), dtype=object)

        for q in kpts.loop(1):
            for kj in kpts.loop(1, mpi=True):
                kb = kpts.member(kpts.wrap_around(kpts[q] + kpts[kj]))
                cou_occ[kj, 0] = np.dot(self.Li[kj].conj(), self.Li[kj].conj().T)
                cou_vir[kb, 0] = np.dot(self.La[kb], self.La[kb].T)
                zeta[q, kb, 0] = cou_occ[kj, 0] * cou_vir[kb, 0]
        zeta[..., 0] /= self.nkpts
        cput1 = lib.logger.timer(self.gw, "zeroth moment", *cput0)

        cou_square = np.zeros((self.nkpts, self.naux, self.naux), dtype=complex)
        for q in kpts.loop(1):
            for kj in kpts.loop(1, mpi=True):
                kb = kpts.member(kpts.wrap_around(kpts[q] + kpts[kj]))
                cou_left[q, kb, 0] = np.eye(self.naux)
                cou_square[q] += np.dot(self.cou[q], (cou_occ[kj, 0] * cou_vir[kb, 0]))

        for i in range(1, self.nmom_max + 1):
            cou_it_add = np.zeros((self.nkpts, self.naux, self.naux), dtype=complex)
            for q in kpts.loop(1):
                for kj in kpts.loop(1, mpi=True):
                    kb = kpts.member(kpts.wrap_around(kpts[q] + kpts[kj]))
                    zeta[q, kb, i] = np.zeros((self.naux, self.naux), dtype=complex)
                    cou_d_left[q, kb, 0] = cou_left[q, kb, 0]
                    cou_d_left[q, kb] = np.roll(cou_d_left[q, kb], 1)
                    cou_left[q, kb, 0] = np.dot(cou_square[q], cou_left[q, kb, 0]) * 2 / self.nkpts

                    ei = self.mo_energy_w[kj][self.mo_occ_w[kj] > 0]
                    ea = self.mo_energy_w[kb][self.mo_occ_w[kb] == 0]
                    cou_ei_max = lib.einsum(
                        "i,Pi,Qi->PQ", ei**i, self.Li[kj].conj(), self.Li[kj].conj()
                    ) * pow(-1, i)
                    cou_ea_max = lib.einsum("a,Pa,Qa->PQ", ea**i, self.La[kb], self.La[kb])

                    cou_d_only[q, kb, i] = cou_ea_max * cou_occ[kj, 0] + cou_ei_max * cou_vir[kb, 0]

                    for j in range(1, i):
                        cou_ei = lib.einsum(
                            "i,Pi,Qi->PQ", ei**j, self.Li[kj].conj(), self.Li[kj].conj()
                        ) * pow(-1, j)
                        cou_ea = lib.einsum(
                            "a,Pa,Qa->PQ", ea ** (i - j), self.La[kb], self.La[kb]
                        ) * binom(i, j)
                        cou_d_only[q, kb, i] += cou_ei * cou_ea
                        if j == (i - 1):
                            cou_it_add[q] += cou_d_only[q, kb, j]
                        else:
                            cou_it_add[q] += np.dot(
                                cou_d_only[q, kb, i - 1 - j], cou_d_left[q, kb, i - j]
                            )
                        zeta[q, kb, i] += (
                            np.dot(cou_d_only[q, kb, j], cou_d_left[q, kb, j]) / self.nkpts
                        )
                cou_it_add[q] = np.dot(self.cou[q], cou_it_add[q])
                cou_it_add[q] *= 2.0
                cou_it_add[q] /= self.nkpts
                for kj in kpts.loop(1, mpi=True):
                    kb = kpts.member(kpts.wrap_around(kpts[q] + kpts[kj]))
                    zeta[q, kb, i] += cou_d_only[q, kb, i] / self.nkpts

                    cou_left[q, kb, 0] += cou_it_add[q]
                    zeta[q, kb, i] += np.dot(zeta[q, kb, 0], cou_left[q, kb, 0])

                    cput1 = lib.logger.timer(self.gw, "moment %d" % i, *cput1)

        return zeta

    def build_se_moments(self, zeta):
        """
        Build the moments of the self-energy via convolution with
        tensor-hypercontraction in k-space.

        Parameters
        ----------
        zeta : numpy.ndarray
            Moments of the density-density response.

        Returns
        -------
        moments_occ : numpy.ndarray
            Moments of the occupied self-energy.
        moments_vir : numpy.ndarray
            Moments of the virtual self-energy.
        """

        cput0 = (lib.logger.process_clock(), lib.logger.perf_counter())
        lib.logger.info(self.gw, "Building self-energy moments")
        lib.logger.debug(self.gw, "Memory usage: %.2f GB", self._memory_usage())

        kpts = self.kpts

        if self.gw.diagonal_se:
            pqchar = pchar = qchar = "p"
            eta_shape = lambda k: (self.mo_energy_g[k].size, self.nmom_max + 1, self.nmo)
        else:
            pqchar, pchar, qchar = "pq", "p", "q"
            eta_shape = lambda k: (self.mo_energy_g[k].size, self.nmom_max + 1, self.nmo, self.nmo)
        eta = np.zeros((self.nkpts, self.nkpts), dtype=object)

        for i in range(self.nmom_max + 1):
            for q in kpts.loop(1):
                zeta_prime = 0
                for kj in kpts.loop(1, mpi=True):
                    kb = kpts.member(kpts.wrap_around(kpts[q] + kpts[kj]))
                    zeta_prime += np.linalg.multi_dot((self.cou[q], zeta[q, kb, i], self.cou[q]))
                zeta_prime *= 2.0
                zeta_prime /= self.nkpts

                for kp in range(self.nkpts):
                    kx = kpts.member(kpts.wrap_around(kpts[kp] - kpts[q]))

                    if not isinstance(eta[kp, q], np.ndarray):
                        eta[kp, q] = np.zeros(eta_shape(kx), dtype=zeta_prime.dtype)

                    for x in range(self.mo_energy_g[kx].size):
                        Lpx = lib.einsum(
                            "Pp,P->Pp", self.integrals.Lp[kp], self.integrals.Lx[kx][:, x]
                        )
                        subscript = f"P{pchar},Q{qchar},PQ->{pqchar}"
                        eta[kp, q][x, i] += lib.einsum(subscript, Lpx, Lpx.conj(), zeta_prime)

        cput1 = lib.logger.timer(self.gw, "rotating DD moments", *cput0)

        # Construct the self-energy moments
        moments_occ, moments_vir = self.convolve(eta)
        cput1 = lib.logger.timer(self.gw, "constructing SE moments", *cput1)

        return moments_occ, moments_vir
