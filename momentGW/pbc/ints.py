"""
Integral helpers with periodic boundary conditions.
"""

from collections import defaultdict

import numpy as np
from pyscf import lib
from pyscf.agf2 import mpi_helper
from pyscf.lib import logger

from momentGW.ints import Integrals


class KIntegrals(Integrals):
    """
    Container for the integrals required for KGW methods.
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
    ):
        Integrals.__init__(
            self,
            with_df,
            mo_coeff,
            mo_occ,
            compression=compression,
            compression_tol=compression_tol * len(kpts),
            store_full=store_full,
        )

        self.kpts = kpts

    def get_compression_metric(self):
        """
        Return the compression metric.
        """

        compression = self._parse_compression()
        if not compression:
            return None

        cput0 = (logger.process_clock(), logger.perf_counter())
        logger.info(self, f"Computing compression metric for {self.__class__.__name__}")

        prod = np.zeros((len(self.kpts), self.naux_full, self.naux_full), dtype=complex)

        # Loop over required blocks
        for key in sorted(compression):
            logger.debug(self, f"Transforming {key} block")
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

            for (q, qpt), (kj, kptj) in self.kpts.loop(2):
                ki = self.kpts.member(self.kpts.wrap_around(qpt - kptj))

                for p0, p1 in lib.prange(0, ni[ki] * nj[kj], self.with_df.blockdim):
                    i0, j0 = divmod(p0, nj[kj])
                    i1, j1 = divmod(p1, nj[kj])

                    Lxy = np.zeros((self.naux_full, p1 - p0), dtype=complex)
                    b1 = 0
                    for block in self.with_df.sr_loop((ki, kj), compact=False):
                        if block[2] == -1:
                            raise NotImplementedError("Low dimensional integrals")
                        block = block[0] + block[1] * 1.0j
                        block = block.reshape(self.naux_full, self.nmo, self.nmo)
                        b0, b1 = b1, b1 + block.shape[0]
                        logger.debug(self, f"  Block [{ki}, {kj}, {p0}:{p1}, {b0}:{b1}]")

                        tmp = lib.einsum(
                            "Lpq,pi,qj->Lij", block, ci[ki][:, i0 : i1 + 1].conj(), cj[kj]
                        )
                        tmp = tmp.reshape(b1 - b0, -1)
                        Lxy[b0:b1] = tmp[:, j0 : j0 + (p1 - p0)]

                    prod[q] += np.dot(Lxy, Lxy.T.conj())

        rot = np.empty((len(self.kpts),), dtype=object)
        if mpi_helper.rank == 0:
            for q, qpt in self.kpts.loop(1):
                e, v = np.linalg.eig(prod[q])
                mask = np.abs(e) > self.compression_tol
                rot[q] = v[:, mask]
        else:
            for q, qpt in self.kpts.loop(1):
                rot[q] = np.zeros((0,), dtype=complex)
        del prod

        for q, qpt in self.kpts.loop(1):
            rot[q] = mpi_helper.bcast(rot[q], root=0)

            if rot[q].shape[-1] == self.naux_full:
                logger.info(self, f"No compression found at q-point {q}")
                rot[q] = None
            else:
                logger.info(
                    self,
                    f"Compressed auxiliary space from {self.naux_full} to {rot[q].shape[-1]} and q-point {q}",
                )
        logger.timer(self, "compression metric", *cput0)

        return rot

    def transform(self, do_Lpq=None, do_Lpx=True, do_Lia=True):
        """
        Initialise the integrals, building:
            - Lpq: the full (aux, MO, MO) array if `store_full`
            - Lpx: the compressed (aux, MO, MO) array
            - Lia: the compressed (aux, occ, vir) array
        """

        # Get the compression metric
        if self._rot is None:
            self._rot = self.get_compression_metric()
        rot = self._rot
        if rot is None:
            eye = np.eye(self.naux_full)
            rot = defaultdict(lambda: eye)
        for q, qpt in self.kpts.loop(1):
            if rot[q] is None:
                rot[q] = np.eye(self.naux_full)

        do_Lpq = self.store_full if do_Lpq is None else do_Lpq
        if not any([do_Lpq, do_Lpx, do_Lia]):
            return

        cput0 = (logger.process_clock(), logger.perf_counter())
        logger.info(self, f"Transforming {self.__class__.__name__}")

        Lpq = np.zeros((len(self.kpts), len(self.kpts)), dtype=object) if do_Lpq else None
        Lpx = np.zeros((len(self.kpts), len(self.kpts)), dtype=object) if do_Lpx else None
        Lia = np.zeros((len(self.kpts), len(self.kpts)), dtype=object) if do_Lia else None
        Lai = np.zeros((len(self.kpts), len(self.kpts)), dtype=object) if do_Lia else None

        for (q, qpt), (kj, kptj) in self.kpts.loop(2):
            ki = self.kpts.member(self.kpts.wrap_around(qpt - kptj))

            # Get the slices on the current process and initialise the arrays
            # o0, o1 = list(mpi_helper.prange(0, self.nmo, self.nmo))[0]
            # p0, p1 = list(mpi_helper.prange(0, self.nmo_g[k], self.nmo_g[k]))[0]
            # q0, q1 = list(mpi_helper.prange(0, self.nocc_w[k] * self.nvir_w[k], self.nocc_w[k] * self.nvir_w[k]))[0]
            o0, o1 = 0, self.nmo
            p0, p1 = 0, self.nmo_g[ki]
            q0, q1 = 0, self.nocc_w[kj] * self.nvir_w[kj]
            Lpq_k = np.zeros((self.naux_full, self.nmo, o1 - o0), dtype=complex) if do_Lpq else None
            Lpx_k = np.zeros((self.naux[q], self.nmo, p1 - p0), dtype=complex) if do_Lpx else None
            Lia_k = np.zeros((self.naux[q], q1 - q0), dtype=complex) if do_Lia else None
            Lai_k = np.zeros((self.naux[q], q1 - q0), dtype=complex) if do_Lia else None

            # Build the integrals blockwise
            b1 = 0
            for block in self.with_df.sr_loop((ki, kj), compact=False):
                if block[2] == -1:
                    raise NotImplementedError("Low dimensional integrals")
                block = block[0] + block[1] * 1.0j
                block = block.reshape(self.naux_full, self.nmo, self.nmo)
                b0, b1 = b1, b1 + block.shape[0]
                logger.debug(self, f"  Block [{ki}, {kj}, {b0}:{b1}]")

                # If needed, rotate the full (L|pq) array
                if do_Lpq:
                    logger.debug(self, f"(L|pq) size: ({self.naux_full}, {self.nmo}, {o1 - o0})")
                    coeffs = (self.mo_coeff[ki], self.mo_coeff[kj][:, o0:o1])
                    Lpq_k[b0:b1] = lib.einsum("Lpq,pi,qj->Lij", block, coeffs[0].conj(), coeffs[1])

                # Compress the block
                block = lib.einsum("L...,LQ->Q...", block, rot[q][b0:b1].conj())

                # Build the compressed (L|px) array
                if do_Lpx:
                    logger.debug(self, f"(L|px) size: ({self.naux[q]}, {self.nmo}, {p1 - p0})")
                    coeffs = (self.mo_coeff[ki], self.mo_coeff_g[kj][:, p0:p1])
                    Lpx_k += lib.einsum("Lpq,pi,qj->Lij", block, coeffs[0].conj(), coeffs[1])

                # Build the compressed (L|ia) array
                if do_Lia:
                    logger.debug(self, f"(L|ia) size: ({self.naux[q]}, {q1 - q0})")
                    i0, a0 = divmod(q0, self.nvir_w[kj])
                    i1, a1 = divmod(q1, self.nvir_w[kj])
                    coeffs = (
                        self.mo_coeff_w[ki][:, i0 : i1 + 1],
                        self.mo_coeff_w[kj][:, self.nocc_w[kj] :],
                    )
                    tmp = lib.einsum("Lpq,pi,qj->Lij", block, coeffs[0].conj(), coeffs[1])
                    tmp = tmp.reshape(self.naux[q], -1)
                    Lia_k += tmp[:, a0 : a0 + (q1 - q0)]

                # Build the compressed (L|ai) array
                if do_Lia:
                    logger.debug(self, f"(L|ai) size: ({self.naux[q]}, {q1 - q0})")
                    i0, a0 = divmod(q0, self.nocc_w[kj])
                    i1, a1 = divmod(q1, self.nocc_w[kj])
                    coeffs = (
                        self.mo_coeff_w[ki][:, self.nocc_w[ki] :],
                        self.mo_coeff_w[kj][:, i0 : i1 + 1],
                    )
                    tmp = lib.einsum("Lpq,pi,qj->Lij", block, coeffs[0].conj(), coeffs[1])
                    tmp = tmp.swapaxes(1, 2)
                    tmp = tmp.reshape(self.naux[q], -1)
                    Lai_k += tmp[:, a0 : a0 + (q1 - q0)]

            if do_Lpq:
                Lpq[ki, kj] = Lpq_k
            if do_Lpx:
                Lpx[ki, kj] = Lpx_k
            if do_Lia:
                Lia[ki, kj] = Lia_k
                Lai[kj, ki] = Lai_k

        if do_Lpq:
            self._blocks["Lpq"] = Lpq
        if do_Lpx:
            self._blocks["Lpx"] = Lpx
        if do_Lia:
            self._blocks["Lia"] = Lia
            self._blocks["Lai"] = Lai

        logger.timer(self, "transform", *cput0)

    def get_j(self, dm, basis="mo"):
        """Build the J matrix."""

        assert basis in ("ao", "mo")

        if not self.store_full or basis == "ao":
            raise NotImplementedError

        vj = np.zeros_like(dm)

        for (ki, kpti), (kk, kptk) in self.kpts.loop(2):
            kj = ki
            kl = self.kpts.conserve(ki, kj, kk)
            buf = lib.einsum("Lpq,pq->L", self.Lpq[kk, kl], dm[kl].conj())
            vj[ki] += lib.einsum("Lpq,L->pq", self.Lpq[ki, kj], buf)

        vj /= len(self.kpts)

        return vj

    def get_k(self, dm, basis="mo"):
        """Build the K matrix."""

        assert basis in ("ao", "mo")

        if not self.store_full or basis == "ao":
            raise NotImplementedError

        vk = np.zeros_like(dm)

        for (ki, kpti), (kk, kptk) in self.kpts.loop(2):
            kj = ki
            kl = self.kpts.conserve(ki, kj, kk)
            buf = np.dot(self.Lpq[ki, kl].reshape(-1, self.nmo), dm[kl])
            buf = buf.reshape(-1, self.nmo, self.nmo).swapaxes(1, 2).reshape(-1, self.nmo)
            vk[ki] += np.dot(buf.T, self.Lpq[kk, kj].reshape(-1, self.nmo)).T.conj()

        vk /= len(self.kpts)

        return vk

    @property
    def Lai(self):
        """
        Return the compressed (aux, W vir, W occ) array.
        """
        return self._blocks["Lai"]

    @property
    def nmo(self):
        """
        Return the number of MOs.
        """
        assert len({c.shape[-1] for c in self.mo_coeff}) == 1
        return self.mo_coeff[0].shape[-1]

    @property
    def nocc(self):
        """
        Return the number of occupied MOs.
        """
        return [np.sum(o > 0) for o in self.mo_occ]

    @property
    def nvir(self):
        """
        Return the number of virtual MOs.
        """
        return [np.sum(o == 0) for o in self.mo_occ]

    @property
    def nmo_g(self):
        """
        Return the number of MOs for the Green's function.
        """
        return [c.shape[-1] for c in self.mo_coeff_g]

    @property
    def nmo_w(self):
        """
        Return the number of MOs for the screened Coulomb interaction.
        """
        return [c.shape[-1] for c in self.mo_coeff_w]

    @property
    def nocc_w(self):
        """
        Return the number of occupied MOs for the screened Coulomb
        interaction.
        """
        return [np.sum(o > 0) for o in self.mo_occ_w]

    @property
    def nvir_w(self):
        """
        Return the number of virtual MOs for the screened Coulomb
        interaction.
        """
        return [np.sum(o == 0) for o in self.mo_occ_w]

    @property
    def naux(self):
        """
        Return the number of auxiliary basis functions, after the
        compression.
        """
        if self._rot is None:
            return [self.naux_full] * len(self.kpts)
        return [c.shape[-1] if c is not None else self.naux_full for c in self._rot]
