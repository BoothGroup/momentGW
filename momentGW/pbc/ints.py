"""
Integral helpers with periodic boundary conditions.
"""

from collections import defaultdict

import h5py
import numpy as np
from pyscf import lib
from pyscf.ao2mo import _ao2mo
from pyscf.pbc import tools
from scipy.linalg import cholesky
from pyscf.pbc.dft.numint import eval_ao
from pyscf.pbc.dft.gen_grid import get_becke_grids

from momentGW import logging, mpi_helper, util
from momentGW.ints import Integrals, require_compression_metric


class KIntegrals(Integrals):
    """
    Container for the integrals required for KGW methods.

    Parameters
    ----------
    with_df : pyscf.pbc.df.DF
        Density fitting object.
    mo_coeff : numpy.ndarray
        Molecular orbital coefficients at each k-point.
    mo_occ : numpy.ndarray
        Molecular orbital occupations at each k-point.
    compression : str, optional
        Compression scheme to use. Default value is `'ia'`. See
        `momentGW.gw` for more details.
    compression_tol : float, optional
        Compression tolerance. Default value is `1e-10`. See
        `momentGW.gw` for more details.
    store_full : bool, optional
        Store the full MO integrals in memory. Default value is
        `False`.
    """

    def __init__(
        self,
        with_df,
        kpts,
        mo_coeff,
        mo_occ,
        compression="ia",
        compression_tol=1e-10,
        store_full=False,
        input_path=None,
    ):
        Integrals.__init__(
            self,
            with_df,
            mo_coeff,
            mo_occ,
            compression=compression,
            compression_tol=compression_tol,
            store_full=store_full,
        )

        # Options
        self.input_path = input_path

        # Attributes
        self.kpts = kpts
        self._madelung = None
        self._naux_full = None
        self._naux = None

    @logging.with_status("Computing compression metric")
    def get_compression_metric(self):
        """
        Return the compression metric.

        Returns
        -------
        rot : numpy.ndarray
            Rotation matrix into the compressed auxiliary space.
        """

        # TODO MPI

        # Get the compression sectors
        compression = self._parse_compression()
        if not compression:
            return None

        # Initialise the inner product matrix
        prod = np.zeros((len(self.kpts)), dtype=object)

        # ao2mo function for both real and complex integrals
        tao = np.empty([], dtype=np.int32)
        ao_loc = self.with_df.cell.ao_loc_nr()

        def _ao2mo_e2(Lpq, mo_coeff, orb_slice, out=None):
            mo_coeff = np.asarray(mo_coeff, order="F")
            if np.iscomplexobj(Lpq):
                out = _ao2mo.r_e2(Lpq, mo_coeff, orb_slice, tao, ao_loc, aosym="s1", out=out)
            else:
                out = _ao2mo.nr_e2(Lpq, mo_coeff, orb_slice, aosym="s1", mosym="s1")
            return out

        # Loop over required blocks
        for key in sorted(compression):
            with logging.with_status(f"{key} sector"):
                # Get the coefficients
                ci, cj = [
                    {
                        "o": [c[:, o > 0] for c, o in zip(self.mo_coeff, self.mo_occ)],
                        "v": [c[:, o == 0] for c, o in zip(self.mo_coeff, self.mo_occ)],
                        "i": [c[:, o > 0] for c, o in zip(self.mo_coeff_w, self.mo_occ_w)],
                        "a": [c[:, o == 0] for c, o in zip(self.mo_coeff_w, self.mo_occ_w)],
                    }[k]
                    for k in key
                ]
                ni = [c.shape[-1] for c in ci]
                nj = [c.shape[-1] for c in cj]

                for q, ki in self.kpts.loop(2):
                    kj = self.kpts.member(self.kpts.wrap_around(self.kpts[ki] - self.kpts[q]))

                    # Build the (L|xy) array
                    Lxy = np.zeros((self.naux_full[q], ni[ki] * nj[kj]), dtype=complex)
                    b1 = 0
                    for block in self.with_df.sr_loop((ki, kj), compact=False):
                        if block[2] == -1:
                            raise NotImplementedError("Low dimensional integrals")
                        block = block[0] + block[1] * 1.0j
                        block = block.reshape(block.shape[0], self.nmo, self.nmo)
                        b0, b1 = b1, b1 + block.shape[0]
                        progress = ki * len(self.kpts) ** 2 + kj * len(self.kpts) + b0
                        progress /= len(self.kpts) ** 2 + block.shape[0]

                        with logging.with_status(f"block [{ki}, {kj}, {b0}:{b1}] ({progress:.1%})"):
                            coeffs = np.concatenate((ci[ki], cj[kj]), axis=1)
                            orb_slice = (0, ni[ki], ni[ki], ni[ki] + nj[kj])
                            _ao2mo_e2(block, coeffs, orb_slice, out=Lxy[b0:b1])

                    # Update the inner product matrix
                    prod[q] += np.dot(Lxy, Lxy.T.conj()) / len(self.kpts)

        # Diagonalise the inner product matrix
        rot = np.empty((len(self.kpts),), dtype=object)
        if mpi_helper.rank == 0:
            for q in self.kpts.loop(1):
                e, v = np.linalg.eigh(prod[q])
                mask = np.abs(e) > self.compression_tol
                rot[q] = v[:, mask]
        else:
            for q in self.kpts.loop(1):
                rot[q] = np.zeros((0,), dtype=complex)
        del prod

        # Print the compression status
        naux_total = sum(r.shape[-1] for r in rot)
        if naux_total == np.sum(self.naux_full):
            logging.write("No compression found for auxiliary space")
            rot = None
        else:
            percent = 100 * naux_total / np.sum(self.naux_full)
            style = logging.rate(percent, 80, 95)
            logging.write(
                f"Compressed auxiliary space from {np.sum(self.naux_full)} to {naux_total} "
                f"([{style}]{percent:.1f}%)[/]"
            )

        return rot

    @require_compression_metric()
    @logging.with_status("Transforming integrals")
    def transform(self, do_Lpq=None, do_Lpx=True, do_Lia=True):
        """
        Transform the integrals in-place.

        Parameters
        ----------
        do_Lpq : bool, optional
            Whether to compute the full ``(aux, MO, MO)`` array. Default
            value is `True` if `store_full` is `True`, `False`
            otherwise.
        do_Lpx : bool, optional
            Whether to compute the compressed ``(aux, MO, MO)`` array.
            Default value is `True`.
        do_Lia : bool, optional
            Whether to compute the compressed ``(aux, occ, vir)`` array.
            Default value is `True`.
        """

        # Get the compression metric
        rot = self._rot
        if rot is None:
            rot = np.zeros(len(self.kpts), dtype=object)
            for q in self.kpts.loop(1):
                rot[q] = np.eye(self.naux[q])

        # Check which arrays to build
        do_Lpq = self.store_full if do_Lpq is None else do_Lpq
        if not any([do_Lpq, do_Lpx, do_Lia]):
            return

        # ao2mo function for both real and complex integrals
        tao = np.empty([], dtype=np.int32)

        def _ao2mo_e2(Lpq, mo_coeff, orb_slice, out=None):
            mo_coeff = np.asarray(mo_coeff, order="F")
            if np.iscomplexobj(Lpq):
                out = _ao2mo.r_e2(Lpq, mo_coeff, orb_slice, tao, ao_loc=None, aosym="s1", out=out)
            else:
                out = _ao2mo.nr_e2(Lpq, mo_coeff, orb_slice, aosym="s1", mosym="s1")
            return out

        # Prepare the outputs
        Lpq = {}
        Lpx = {}
        Lia = {}
        Lai = {}

        for q in self.kpts.loop(1):
            for ki in self.kpts.loop(1, mpi=True):
                kj = self.kpts.member(self.kpts.wrap_around(self.kpts[q] + self.kpts[ki]))

                # Get the slices on the current process and initialise
                # the arrays
                Lpq_k = (
                    np.zeros((self.naux_full[q], self.nmo, self.nmo), dtype=complex)
                    if do_Lpq
                    else None
                )
                Lpx_k = (
                    np.zeros((self.naux[q], self.nmo, self.nmo_g[kj]), dtype=complex)
                    if do_Lpx
                    else None
                )
                Lia_k = (
                    np.zeros((self.naux[q], self.nocc_w[ki] * self.nvir_w[kj]), dtype=complex)
                    if do_Lia
                    else None
                )
                Lai_k = (
                    np.zeros((self.naux[q], self.nocc_w[ki] * self.nvir_w[kj]), dtype=complex)
                    if do_Lia
                    else None
                )

                # Build the integrals blockwise
                b1 = 0
                for block in self.with_df.sr_loop((ki, kj), compact=False):  # TODO lock I/O
                    if block[2] == -1:
                        raise NotImplementedError("Low dimensional integrals")
                    block = block[0] + block[1] * 1.0j
                    b0, b1 = b1, b1 + block.shape[0]
                    progress = ki * len(self.kpts) ** 2 + kj * len(self.kpts) + b0
                    progress /= len(self.kpts) ** 2 + self.naux[q]

                    with logging.with_status(f"block [{ki}, {kj}, {b0}:{b1}] ({progress:.1%})"):
                        # If needed, rotate the full (L|pq) array
                        if do_Lpq:
                            coeffs = np.concatenate((self.mo_coeff[ki], self.mo_coeff[kj]), axis=1)
                            orb_slice = (0, self.nmo, self.nmo, self.nmo + self.nmo)
                            _ao2mo_e2(block, coeffs, orb_slice, out=Lpq_k[b0:b1])

                        # Compress the block
                        block_comp = util.einsum("L...,LQ->Q...", block, rot[q][b0:b1].conj())

                        # Build the compressed (L|px) array
                        if do_Lpx:
                            coeffs = np.concatenate(
                                (self.mo_coeff[ki], self.mo_coeff_g[kj]), axis=1
                            )
                            orb_slice = (0, self.nmo, self.nmo, self.nmo + self.nmo_g[kj])
                            tmp = _ao2mo_e2(block_comp, coeffs, orb_slice)
                            Lpx_k += tmp.reshape(Lpx_k.shape)

                        # Build the compressed (L|ia) array
                        if do_Lia:
                            coeffs = np.concatenate(
                                (
                                    self.mo_coeff_w[ki][:, self.mo_occ_w[ki] > 0],
                                    self.mo_coeff_w[kj][:, self.mo_occ_w[kj] == 0],
                                ),
                                axis=1,
                            )
                            orb_slice = (
                                0,
                                self.nocc_w[ki],
                                self.nocc_w[ki],
                                self.nocc_w[ki] + self.nvir_w[kj],
                            )
                            tmp = _ao2mo_e2(block_comp, coeffs, orb_slice)
                            Lia_k += tmp.reshape(Lia_k.shape)

                # Store the blocks
                if do_Lpq:
                    Lpq[ki, kj] = Lpq_k
                if do_Lpx:
                    Lpx[ki, kj] = Lpx_k
                if do_Lia:
                    Lia[ki, kj] = Lia_k
                else:
                    continue

                # Inverse q for ki <-> kj
                invq = self.kpts.member(self.kpts.wrap_around(-self.kpts[q]))

                # Build the integrals blockwise
                b1 = 0
                for block in self.with_df.sr_loop((kj, ki), compact=False):  # TODO lock I/O
                    if block[2] == -1:
                        raise NotImplementedError("Low dimensional integrals")
                    block = block[0] + block[1] * 1.0j
                    b0, b1 = b1, b1 + block.shape[0]
                    progress = ki * len(self.kpts) ** 2 + kj * len(self.kpts) + b0
                    progress /= len(self.kpts) ** 2 + self.naux_full[invq]

                    with logging.with_status(f"block [{ki}, {kj}, {b0}:{b1}] ({progress:.1%})"):
                        # Compress the block
                        block_comp = util.einsum("L...,LQ->Q...", block, rot[invq][b0:b1].conj())

                        # Build the compressed (L|ai) array
                        coeffs = np.concatenate(
                            (
                                self.mo_coeff_w[kj][:, self.mo_occ_w[kj] == 0],
                                self.mo_coeff_w[ki][:, self.mo_occ_w[ki] > 0],
                            ),
                            axis=1,
                        )
                        orb_slice = (
                            0,
                            self.nvir_w[kj],
                            self.nvir_w[kj],
                            self.nvir_w[kj] + self.nocc_w[ki],
                        )
                        tmp = _ao2mo_e2(block_comp, coeffs, orb_slice)
                        tmp = tmp.reshape(self.naux[invq], self.nvir_w[kj], self.nocc_w[ki])
                        tmp = tmp.swapaxes(1, 2)
                        Lai_k += tmp.reshape(Lai_k.shape)

                Lai[ki, kj] = Lai_k

        # Store the arrays
        if do_Lpq:
            self._blocks["Lpq"] = Lpq
        if do_Lpx:
            self._blocks["Lpx"] = Lpx
        if do_Lia:
            self._blocks["Lia"] = Lia
            self._blocks["Lai"] = Lai

    def get_cderi_from_thc(self):
        """
        Build CDERIs using THC integrals imported from a h5py file.
        It must contain a 'collocation_matrix' and a 'coulomb_matrix'.
        """

        if self.input_path is None:
            raise ValueError(
                "A file path containing the THC integrals is needed for the THC implementation"
            )
        if "thc" not in self.input_path.lower():
            raise ValueError("File path must contain 'thc' or 'THC' for THC implementation")

        thc_eri = h5py.File(self.input_path, "r")
        kpts_imp = np.array(thc_eri["kpts"])

        if self.kpts != kpts_imp:
            raise ValueError("Different kpts imported to those from PySCF")

        Lpx = {}
        Lia = {}
        Lai = {}
        self._naux = [np.array(thc_eri["coulomb_matrix"])[0, ..., 0].shape[0]] * len(self.kpts)
        for q in self.kpts.loop(1):
            for ki in self.kpts.loop(1):
                kj = self.kpts.member(self.kpts.wrap_around(self.kpts[q] + self.kpts[ki]))

                Lpx_k = np.zeros((self.naux[q], self.nmo, self.nmo_g[kj]), dtype=complex)
                Lia_k = np.zeros((self.naux[q], self.nocc_w[ki] * self.nvir_w[kj]), dtype=complex)
                Lai_k = np.zeros((self.naux[q], self.nocc_w[ki] * self.nvir_w[kj]), dtype=complex)

                cou = np.asarray(thc_eri["coulomb_matrix"])[q, ..., 0]
                coll_ki = np.asarray(thc_eri["collocation_matrix"])[0, ki, ..., 0]
                coll_kj = np.asarray(thc_eri["collocation_matrix"])[0, kj, ..., 0]
                cholesky_cou = cholesky(cou, lower=True)

                block = util.einsum("Pp,Pq,PQ->Qpq", coll_ki.conj(), coll_kj, cholesky_cou)

                coeffs = (self.mo_coeff[ki], self.mo_coeff_g[kj])
                Lpx_k += util.einsum("Lpq,pi,qj->Lij", block, coeffs[0].conj(), coeffs[1])
                coeffs = (
                    self.mo_coeff_w[ki][:, self.mo_occ_w[ki] > 0],
                    self.mo_coeff_w[kj][:, self.mo_occ_w[kj] == 0],
                )
                tmp = util.einsum("Lpq,pi,qj->Lij", block, coeffs[0].conj(), coeffs[1])
                tmp = tmp.reshape(self.naux[q], -1)
                Lia_k += tmp

                Lpx[ki, kj] = Lpx_k
                Lia[ki, kj] = Lia_k

                invq = self.kpts.member(self.kpts.wrap_around(-self.kpts[q]))

                block_switch = util.einsum("Pp,Pq,PQ->Qpq", coll_kj.conj(), coll_ki, cholesky_cou)

                coeffs = (
                    self.mo_coeff_w[kj][:, self.mo_occ_w[kj] == 0],
                    self.mo_coeff_w[ki][:, self.mo_occ_w[ki] > 0],
                )
                tmp = util.einsum("Lpq,pi,qj->Lij", block_switch, coeffs[0].conj(), coeffs[1])
                tmp = tmp.swapaxes(1, 2)
                tmp = tmp.reshape(self.naux[invq], -1)
                Lai_k += tmp

                Lai[ki, kj] = Lai_k

        self._blocks["Lpx"] = Lpx
        self._blocks["Lia"] = Lia
        self._blocks["Lai"] = Lai

    def get_j(self, dm, basis="mo", other=None):
        """Build the J matrix.

        Parameters
        ----------
        dm : numpy.ndarray
            Density matrix at each k-point.
        basis : str, optional
            Basis in which to build the J matrix. One of
            `("ao", "mo")`. Default value is `"mo"`.
        other : Integrals, optional
            Integrals object for the ket side. Allows inheritence for
            mixed-spin evaluations. If `None`, use `self`. Default
            value is `None`.

        Returns
        -------
        vj : numpy.ndarray
            J matrix.

        Notes
        -----
        The contraction is
        `J[p, q] = self[p, q] * other[r, s] * dm[r, s]`, and the
        bases must reflect shared indices.
        """

        # Check the input
        assert basis in ("ao", "mo")

        # Get the other integrals
        if other is None:
            other = self

        # Initialise the J matrix
        vj = np.zeros_like(dm, dtype=complex)

        if self.store_full and basis == "mo":
            # Constuct J using the full MO basis integrals
            buf = 0.0
            for kk in self.kpts.loop(1, mpi=True):
                buf += util.einsum("Lpq,pq->L", other.Lpq[kk, kk], dm[kk].conj())

            buf = mpi_helper.allreduce(buf)

            for ki in self.kpts.loop(1, mpi=True):
                vj[ki] += util.einsum("Lpq,L->pq", self.Lpq[ki, ki], buf)

            vj = mpi_helper.allreduce(vj)

        else:
            # Transform the density into the AO basis
            if basis == "mo":
                dm = util.einsum("kij,kpi,kqj->kpq", dm, other.mo_coeff, np.conj(other.mo_coeff))

            buf = np.zeros((self.naux_full[0],), dtype=complex)

            for kk in self.kpts.loop(1, mpi=True):
                b1 = 0
                for block in self.with_df.sr_loop((kk, kk), compact=False):  # TODO lock I/O
                    if block[2] == -1:
                        raise NotImplementedError("Low dimensional integrals")
                    block = block[0] + block[1] * 1.0j
                    block = block.reshape(self.naux_full, self.nmo, self.nmo)
                    b0, b1 = b1, b1 + block.shape[0]
                    buf[b0:b1] += util.einsum("Lpq,pq->L", block, dm[kk].conj())

            buf = mpi_helper.allreduce(buf)

            for ki in self.kpts.loop(1, mpi=True):
                b1 = 0
                for block in self.with_df.sr_loop((ki, ki), compact=False):
                    if block[2] == -1:
                        raise NotImplementedError("Low dimensional integrals")
                    block = block[0] + block[1] * 1.0j
                    block = block.reshape(self.naux_full, self.nmo, self.nmo)
                    b0, b1 = b1, b1 + block.shape[0]
                    vj[ki] += util.einsum("Lpq,L->pq", block, buf[b0:b1])

            vj = mpi_helper.allreduce(vj)

            # Transform the J matrix back to the MO basis
            if basis == "mo":
                vj = util.einsum("kpq,kpi,kqj->kij", vj, np.conj(self.mo_coeff), self.mo_coeff)

        vj /= len(self.kpts)

        return vj

    @logging.with_timer("K matrix")
    @logging.with_status("Building K matrix")
    def get_k(self, dm, basis="mo", ewald=False):
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
        The contraction is
        `K[p, q] = self[r, q] * self[p, r] * dm[q, s]`, and the
        bases must reflect shared indices.
        """

        # Check the input
        assert basis in ("ao", "mo")

        # Initialise the K matrix
        vk = np.zeros_like(dm, dtype=complex)

        if self.store_full and basis == "mo":
            # Constuct K using the full MO basis integrals
            for p0, p1 in lib.prange(0, np.max(self.naux_full), 240):
                buf = np.zeros((len(self.kpts), len(self.kpts)), dtype=object)
                for ki in self.kpts.loop(1, mpi=True):
                    for kk in self.kpts.loop(1):
                        q = self.kpts.member(self.kpts.wrap_around(self.kpts[kk] - self.kpts[ki]))
                        if p1 > self.naux_full[q]:
                            p1 = self.naux_full[q]
                        buf[kk, ki] = util.einsum("Lpq,qr->Lrp", self.Lpq[ki, kk][p0:p1], dm[kk])

                buf = mpi_helper.allreduce(buf)

                for ki in self.kpts.loop(1):
                    for kk in self.kpts.loop(1, mpi=True):
                        vk[ki] += util.einsum("Lrp,Lrs->ps", buf[kk, ki], self.Lpq[kk, ki][p0:p1])

            vk = mpi_helper.allreduce(vk)

        else:
            # Transform the density into the AO basis
            if basis == "mo":
                dm = util.einsum("kij,kpi,kqj->kpq", dm, self.mo_coeff, np.conj(self.mo_coeff))

            for q in self.kpts.loop(1):
                buf = np.zeros(
                    (len(self.kpts), self.naux_full[q], self.nmo, self.nmo), dtype=complex
                )
                for ki in self.kpts.loop(1, mpi=True):
                    b1 = 0
                    for block in self.with_df.sr_loop((ki, kk), compact=False):
                        if block[2] == -1:
                            raise NotImplementedError("Low dimensional integrals")
                        block = block[0] + block[1] * 1.0j
                        block = block.reshape(block.shape[0], self.nmo, self.nmo)
                        b0, b1 = b1, b1 + block.shape[0]
                        buf[ki, b0:b1] = util.einsum("Lpq,qr->Lrp", block, dm[kk])

                buf = mpi_helper.allreduce(buf)

                for ki in self.kpts.loop(1, mpi=True):
                    b1 = 0
                    for block in self.with_df.sr_loop((kk, ki), compact=False):
                        if block[2] == -1:
                            raise NotImplementedError("Low dimensional integrals")
                        block = block[0] + block[1] * 1.0j
                        block = block.reshape(block.shape[0], self.nmo, self.nmo)
                        b0, b1 = b1, b1 + block.shape[0]
                        vk[ki] += util.einsum("Lrp,Lrs->ps", buf[ki, b0:b1], block)

            vk = mpi_helper.allreduce(vk)

            # Transform the K matrix back to the MO basis
            if basis == "mo":
                vk = util.einsum("kpq,kpi,kqj->kij", vk, np.conj(self.mo_coeff), self.mo_coeff)

        vk /= len(self.kpts)

        if ewald:
            vk += self.get_ewald(dm, basis=basis)

        return vk

    @logging.with_timer("Ewald matrix")
    @logging.with_status("Building Ewald matrix")
    def get_ewald(self, dm, basis="mo"):
        """Build the Ewald exchange divergence matrix.

        Parameters
        ----------
        dm : numpy.ndarray
            Density matrix at each k-point.
        basis : str, optional
            Basis in which to build the K matrix. One of
            `("ao", "mo")`. Default value is `"mo"`.

        Returns
        -------
        ew : numpy.ndarray
            Ewald exchange divergence matrix at each k-point.
        """

        # Check the input
        assert basis in ("ao", "mo")

        # Get the overlap matrix
        if basis == "mo":
            ovlp = defaultdict(lambda: np.eye(self.nmo))
        else:
            ovlp = self.with_df.cell.pbc_intor("int1e_ovlp", hermi=1, kpts=self.kpts._kpts)

        # Initialise the Ewald matrix
        ew = util.einsum("kpq,kpi,kqj->kij", dm, ovlp.conj(), ovlp)

        return ew

    def get_jk(self, dm, **kwargs):
        """Build the J and K matrices.

        Returns
        -------
        vj : numpy.ndarray
            J matrix at each k-point.
        vk : numpy.ndarray
            K matrix at each k-point.

        Notes
        -----
        See `get_j` and `get_k` for more information.
        """
        return super().get_jk(dm, **kwargs)

    def get_veff(self, dm, j=None, k=None, **kwargs):
        """Build the effective potential.

        Returns
        -------
        veff : numpy.ndarray
            Effective potential at each k-point.
        j : numpy.ndarray, optional
            J matrix at each k-point. If `None`, compute it. Default
            value is `None`.
        k : numpy.ndarray, optional
            K matrix at each k-point. If `None`, compute it. Default
            value is `None`.

        Notes
        -----
        See `get_jk` for more information.
        """
        return super().get_veff(dm, j=j, k=k, **kwargs)

    def get_fock(self, dm, h1e, **kwargs):
        """Build the Fock matrix.

        Parameters
        ----------
        dm : numpy.ndarray
            Density matrix at each k-point.
        h1e : numpy.ndarray
            Core Hamiltonian matrix at each k-point.
        **kwargs : dict, optional
            Additional keyword arguments for `get_jk`.

        Returns
        -------
        fock : numpy.ndarray
            Fock matrix at each k-point.

        Notes
        -----
        See `get_jk` for more information. The basis of `h1e` must be
        the same as `dm`.
        """
        return super().get_fock(dm, h1e, **kwargs)

    def get_q_ij(self,q,mo_energy_w):
        cell = self.with_df.cell
        kpts = self.kpts
        coords, weights = get_becke_grids(cell, level=5)
        qij = np.zeros((len(kpts), self.nocc_w[0] * self.nvir_w[0]), dtype=complex)
        for ki in kpts.loop(1):
            psi_all = eval_ao(cell, coords, kpt=kpts[ki], deriv=1)
            psi = psi_all[0]
            psi_div = psi_all[1:4]
            braket = lib.einsum(
                "w,pw,dwq->dpq", weights, psi.T.conj(), psi_div
            )

            num_ao = -1.0j * lib.einsum(
                "d,dpq->pq", q, braket
            )
            num_mo = np.linalg.multi_dot(
                (self.mo_coeff_w[ki][:, self.mo_occ_w[ki] > 0].T.conj(),
                 num_ao,
                 self.mo_coeff_w[ki][:, self.mo_occ_w[ki] == 0])
            )
            den = 1/(mo_energy_w[ki][self.mo_occ_w[ki] == 0, None] - mo_energy_w[ki][None,self.mo_occ_w[ki] > 0])
            qij[ki] = (den.T * num_mo).flatten()

        return qij


    def reciprocal_lattice(self):
        """
        Return the reciprocal lattice vectors.
        """
        return self.with_df.cell.reciprocal_vectors()

    @property
    def madelung(self):
        """
        Return the Madelung constant for the lattice.
        """
        if self._madelung is None:
            self._madeling = tools.pbc.madelung(self.with_df.cell, self.kpts._kpts)
        return self._madelung

    @property
    def Lai(self):
        """Get the full uncompressed ``(aux, MO, MO)`` integrals."""
        return self._blocks["Lai"]

    @property
    def nmo(self):
        """Get the number of MOs."""
        assert len({c.shape[-1] for c in self.mo_coeff}) == 1
        return self.mo_coeff[0].shape[-1]

    @property
    def nocc(self):
        """Get the number of occupied MOs."""
        return [np.sum(o > 0) for o in self.mo_occ]

    @property
    def nvir(self):
        """Get the number of virtual MOs."""
        return [np.sum(o == 0) for o in self.mo_occ]

    @property
    def nmo_g(self):
        """Get the number of MOs for the Green's function."""
        return [c.shape[-1] for c in self.mo_coeff_g]

    @property
    def nmo_w(self):
        """
        Get the number of MOs for the screened Coulomb interaction.
        """
        return [c.shape[-1] for c in self.mo_coeff_w]

    @property
    def nocc_w(self):
        """
        Get the number of occupied MOs for the screened Coulomb
        interaction.
        """
        return [np.sum(o > 0) for o in self.mo_occ_w]

    @property
    def nvir_w(self):
        """
        Get the number of virtual MOs for the screened Coulomb
        interaction.
        """
        return [np.sum(o == 0) for o in self.mo_occ_w]

    @property
    def naux(self):
        """
        Get the number of auxiliary basis functions, after the
        compression.
        """
        if self._rot is None:
            if self._naux is not None:
                return self._naux
            else:
                return self.naux_full

        return [
            c.shape[-1] if c is not None else self.naux_full[i] for i, c in enumerate(self._rot)
        ]

    @property
    def naux_full(self):
        """
        Get the number of auxiliary basis functions, before the
        compression.
        """
        if self._naux_full is None:
            self._naux_full = np.zeros(len(self.kpts), dtype=int)
            for ki in self.kpts.loop(1):
                for block in self.with_df.sr_loop((0, ki), compact=False):
                    if block[2] == -1:
                        raise NotImplementedError("Low dimensional integrals")
                    self._naux_full[ki] += block[0].shape[0]

        return self._naux_full
