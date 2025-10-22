"""Tensor hyper-contraction with periodic boundary conditions."""

import h5py
import numpy as np
from scipy.special import binom

from momentGW import logging, util
from momentGW.pbc.ints import KIntegrals as DFKIntegrals
from momentGW.pbc.tda import dTDA as DFdTDA
from momentGW.thc import Integrals
from momentGW.thc import dTDA as MoldTDA


class KIntegrals(Integrals, DFKIntegrals):
    """Container for the tensor-hypercontracted integrals required for GW methods with periodic
    boundary conditions.

    Parameters
    ----------
    with_df : pyscf.pbc.df.DF
        Density fitting object.
    mo_coeff : numpy.ndarray
        Molecular orbital coefficients at each k-point.
    mo_occ : numpy.ndarray
        Molecular orbital occupations at each k-point.
    file_path : str, optional
        Path to the HDF5 file containing the integrals. Default value is
        `None`.
    """

    def __init__(
        self,
        with_df,
        kpts,
        mo_coeff,
        mo_occ,
        file_path=None,
        store_full=False,
    ):
        Integrals.__init__(
            self,
            with_df,
            mo_coeff,
            mo_occ,
            file_path=file_path,
        )

        # Parameters
        self.kpts = kpts
        self.store_full = store_full

        # Options
        self.compression = None

    def import_thc_components(self):
        """Import a HDF5 file containing a dictionary.

        The keys
        `"collocation_matrix"` and a `"coulomb_matrix"` must exist, with
        shapes ``(MO, aux)`` and ``(aux, aux)``, respectively.
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
            coll[ki] = np.array(thc_eri["collocation_matrix"])[0, ki, ..., 0]

        self._blocks["coll"] = coll
        self._blocks["cou"] = cou

    @logging.with_status("Transforming integrals")
    def transform(self, do_Lpq=True, do_Lpx=True, do_Lia=True):
        """Transform the integrals in-place.

        Parameters
        ----------
        do_Lpq : bool, optional
            Whether the ``(aux, MO, MO)`` array is required. In THC,
            this requires the `Lp` array. Default value is `True`.
        do_Lpx : bool, optional
            Whether the ``(aux, MO, MO)`` array is required. In THC,
            this requires the `Lx` array. Default value is `True`.
        do_Lia : bool, optional
            Whether the ``(aux, occ, vir)`` array is required. In THC,
            this requires the `Li` and `La` arrays. Default value is
            `True`.
        """

        # Check if any arrays are required
        if not any([do_Lpq, do_Lpx, do_Lia]):
            return

        # Import THC components
        if self.coll is None and self.cou is None:
            self.import_thc_components()

        Lp = {}
        Lx = {}
        Li = {}
        La = {}

        do_Lpq = self.store_full if do_Lpq is None else do_Lpq

        for ki in range(self.nkpts):
            # Transform the (L|pq) array
            if do_Lpq:
                Lp[ki] = util.einsum("Lp,pq->Lq", self.coll[ki], self.mo_coeff[ki])

            # Transform the (L|px) array
            if do_Lpx:
                Lx[ki] = util.einsum("Lp,pq->Lq", self.coll[ki], self.mo_coeff_g[ki])

            # Transform the (L|ia) and (L|ai) arrays
            if do_Lia:
                ci = self.mo_coeff_w[ki][:, self.mo_occ_w[ki] > 0]
                ca = self.mo_coeff_w[ki][:, self.mo_occ_w[ki] == 0]

                Li[ki] = util.einsum("Lp,pi->Li", self.coll[ki], ci)
                La[ki] = util.einsum("Lp,pa->La", self.coll[ki], ca)

        if do_Lpq:
            self._blocks["Lp"] = Lp
        if do_Lpx:
            self._blocks["Lx"] = Lx
        if do_Lia:
            self._blocks["Li"] = Li
            self._blocks["La"] = La

    @logging.with_timer("J matrix")
    @logging.with_status("Building J matrix")
    def get_j(self, dm, basis="mo"):
        """Build the J matrix.

        Parameters
        ----------
        dm : numpy.ndarray
            Density matrix at each k-point.
        basis : str, optional
            Basis in which to build the J matrix. One of
            `("ao", "mo")`. Default value is `"mo"`.

        Returns
        -------
        vj : numpy.ndarray
            J matrix at each k-point.

        Notes
        -----
        The basis of `dm` must be the same as `basis`.
        """

        # Check the input
        assert basis in ("ao", "mo")

        # Get the components
        vj = np.zeros_like(dm, dtype=complex)
        if basis == "ao":
            if self.coll is None and self.cou is None:
                self.import_thc_components()
            Lp = self.coll
            cou = self.cou
        else:
            Lp = self.Lp
            cou = self.cou

        buf = 0.0
        for ki in range(self.nkpts):
            tmp = util.einsum("pq,Kp,Kq->K", dm[ki], Lp[ki], Lp[ki].conj())
            tmp = util.einsum("K,KL->L", tmp, cou[0])
            buf += tmp

        buf /= self.nkpts

        for kj in range(self.nkpts):
            vj[kj] = util.einsum("L,Lr,Ls->rs", buf, Lp[kj].conj(), Lp[kj])

        return vj

    @logging.with_timer("K matrix")
    @logging.with_status("Building K matrix")
    def get_k(self, dm, basis="mo"):
        """Build the K matrix.

        Parameters
        ----------
        dm : numpy.ndarray
            Density matrix at each k-point.
        basis : str, optional
            Basis in which to build the K matrix. One of
            `("ao", "mo")`. Default value is `"mo"`.

        Returns
        -------
        vk : numpy.ndarray
            K matrix at each k-point.

        Notes
        -----
        The basis of `dm` must be the same as `basis`.
        """

        # Check the input
        assert basis in ("ao", "mo")

        # Get the components
        vk = np.zeros_like(dm, dtype=complex)
        if basis == "ao":
            if self.coll is None and self.cou is None:
                self.import_thc_components()
            Lp = self.coll
            cou = self.cou
        else:
            Lp = self.Lp
            cou = self.cou

        buf = np.zeros((self.nkpts, self.nkpts, self.naux, self.naux), dtype=complex)
        for ki in range(self.nkpts):
            for kk in range(self.nkpts):
                tmp = util.einsum("pq,Kp->Kq", dm[kk], Lp[kk].conj())
                tmp = util.einsum("Kq,Lq->KL", tmp, Lp[kk])
                kb = self.kpts.member(self.kpts.wrap_around(self.kpts[ki] + self.kpts[kk]))
                buf[ki, kk] = util.einsum("KL,KL->KL", tmp, cou[kb])

        buf /= self.nkpts
        for ki in range(self.nkpts):
            for kk in range(self.nkpts):
                tmp = util.einsum("KL,Ks->Ls", buf[ki, kk], Lp[ki].conj())
                vk[ki] += util.einsum("Ls,Lr->rs", tmp, Lp[ki])

        return vk

    @property
    def nkpts(self):
        """Get the number of k-points."""
        return len(self.kpts)

    @property
    def naux(self):
        """Get the number of auxiliary basis functions."""
        return self.cou[0].shape[0]


class dTDA(MoldTDA, DFdTDA):
    """Compute the self-energy moments using dTDA with tensor hyper-contraction and periodic
    boundary conditions.

    Parameters
    ----------
    gw : BaseKGW
        GW object.
    nmom_max : int
        Maximum moment number to calculate.
    integrals : KIntegrals
        Density-fitted integrals.
    mo_energy : numpy.ndarray or tuple of numpy.ndarray, optional
        Molecular orbital energies at each k-point. If a tuple is passed,
        the first element corresponds to the Green's function basis and
        the second to the screened Coulomb interaction. Default value is
        that of `gw.mo_energy`.
    mo_occ : numpy.ndarray or tuple of numpy.ndarray, optional
        Molecular orbital occupancies at each k-point. If a tuple is
        passed, the first element corresponds to the Green's function basis
        and the second to the screened Coulomb interaction. Default value
        is that of `gw.mo_occ`.
    """

    @logging.with_timer("Density-density moments")
    @logging.with_status("Constructing density-density moments")
    def build_dd_moments(self):
        """Build the moments of the density-density response.

        Returns
        -------
        moments : numpy.ndarray
            Moments of the density-density response at each k-point.

        Notes
        -----
        Unlike the standard `momentGW.tda` implementation, this method
        scales as :math:`O(N^3)` with system size instead of
        :math:`O(N^4)`.
        """

        zeta = np.zeros((self.nkpts, self.nkpts, self.nmom_max + 1), dtype=object)

        kpts = self.kpts
        naux = self.naux
        cou_occ = np.zeros((self.nkpts, 1), dtype=object)
        cou_vir = np.zeros((self.nkpts, 1), dtype=object)

        cou_d_left = np.zeros((self.nkpts, self.nkpts, self.nmom_max + 1), dtype=object)
        cou_d_only = np.zeros((self.nkpts, self.nkpts, self.nmom_max + 1), dtype=object)
        cou_left = np.zeros((self.nkpts, self.nkpts, 1), dtype=object)

        for q in kpts.loop(1):
            for kj in kpts.loop(1):
                kb = kpts.member(kpts.wrap_around(kpts[q] + kpts[kj]))
                cou_occ[kj, 0] = np.dot(self.Li[kj].conj(), self.Li[kj].conj().T)
                cou_vir[kb, 0] = np.dot(self.La[kb], self.La[kb].T)
                zeta[q, kb, 0] = cou_occ[kj, 0] * cou_vir[kb, 0]
        zeta[..., 0] /= self.nkpts

        cou_square = np.zeros((self.nkpts, naux, naux), dtype=complex)
        for q in kpts.loop(1):
            for kj in kpts.loop(1):
                kb = kpts.member(kpts.wrap_around(kpts[q] + kpts[kj]))
                cou_left[q, kb, 0] = np.eye(naux)
                cou_square[q] += np.dot(self.cou[q], (cou_occ[kj, 0] * cou_vir[kb, 0]))

        for i in range(1, self.nmom_max + 1):
            cou_it_add = np.zeros((self.nkpts, naux, naux), dtype=complex)
            for q in kpts.loop(1):
                for kj in kpts.loop(1):
                    kb = kpts.member(kpts.wrap_around(kpts[q] + kpts[kj]))
                    zeta[q, kb, i] = np.zeros((naux, naux), dtype=complex)
                    cou_d_left[q, kb, 0] = cou_left[q, kb, 0]
                    cou_d_left[q, kb] = np.roll(cou_d_left[q, kb], 1)
                    cou_left[q, kb, 0] = np.dot(cou_square[q], cou_left[q, kb, 0]) * 2 / self.nkpts

                    ei = self.mo_energy_w[kj][self.mo_occ_w[kj] > 0]
                    ea = self.mo_energy_w[kb][self.mo_occ_w[kb] == 0]
                    cou_ei_max = util.einsum(
                        "i,Pi,Qi->PQ", ei**i, self.Li[kj].conj(), self.Li[kj].conj()
                    ) * pow(-1, i)
                    cou_ea_max = util.einsum("a,Pa,Qa->PQ", ea**i, self.La[kb], self.La[kb])

                    cou_d_only[q, kb, i] = cou_ea_max * cou_occ[kj, 0] + cou_ei_max * cou_vir[kb, 0]

                    for j in range(1, i):
                        cou_ei = util.einsum(
                            "i,Pi,Qi->PQ", ei**j, self.Li[kj].conj(), self.Li[kj].conj()
                        ) * pow(-1, j)
                        cou_ea = util.einsum(
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
                for kj in kpts.loop(1):
                    kb = kpts.member(kpts.wrap_around(kpts[q] + kpts[kj]))
                    zeta[q, kb, i] += cou_d_only[q, kb, i] / self.nkpts

                    cou_left[q, kb, 0] += cou_it_add[q]
                    zeta[q, kb, i] += np.dot(zeta[q, kb, 0], cou_left[q, kb, 0])

        return zeta

    @logging.with_timer("Self-energy moments")
    @logging.with_status("Constructing self-energy moments")
    def build_se_moments(self, zeta):
        """Build the moments of the self-energy via convolution.

        Parameters
        ----------
        zeta : numpy.ndarray
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

        # Get the moments in (aux|aux) and rotate to (mo|mo)
        for i in range(self.nmom_max + 1):
            for q in kpts.loop(1):
                zeta_prime = 0
                for kj in kpts.loop(1):
                    kb = kpts.member(kpts.wrap_around(kpts[q] + kpts[kj]))
                    zeta_prime += np.linalg.multi_dot((self.cou[q], zeta[q, kb, i], self.cou[q]))
                zeta_prime *= 2.0
                zeta_prime /= self.nkpts

                for kp in range(self.nkpts):
                    kx = kpts.member(kpts.wrap_around(kpts[kp] - kpts[q]))

                    if not isinstance(eta[kp, q], np.ndarray):
                        eta[kp, q] = np.zeros(eta_shape(kx), dtype=zeta_prime.dtype)

                    for x in range(self.mo_energy_g[kx].size):
                        Lpx = util.einsum(
                            "Pp,P->Pp", self.integrals.Lp[kp], self.integrals.Lx[kx][:, x]
                        )
                        subscript = f"P{pchar},Q{qchar},PQ->{pqchar}"
                        eta[kp, q][x, i] += util.einsum(subscript, Lpx, Lpx.conj(), zeta_prime)

        # Construct the self-energy moments
        moments_occ, moments_vir = self.convolve(eta)

        return moments_occ, moments_vir
