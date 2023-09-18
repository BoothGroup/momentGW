"""
Integral helpers.
"""

import contextlib
import types

import numpy as np
from pyscf import lib
from pyscf.agf2 import mpi_helper
from pyscf.lib import logger


@contextlib.contextmanager
def patch_df_loop(with_df):
    """
    Context manager for monkey patching PySCF's density fitting objects
    to loop over blocks of the auxiliary functions distributed over MPI.

    Parameters
    ----------
    with_df : pyscf.df.DF
        Density fitting object.

    Yields
    ------
    with_df : pyscf.df.DF
        Density fitting object with monkey patched `loop` method.
    """

    def prange(self, start, stop, end):
        yield from mpi_helper.prange(start, stop, end)

    pre_patch = with_df.prange
    setattr(with_df, "prange", types.MethodType(prange, with_df))

    yield with_df

    setattr(with_df, "prange", pre_patch)


class Integrals:
    """
    Container for the density-fitted integrals required for GW methods.

    Parameters
    ----------
    with_df : pyscf.df.DF
        Density fitting object.
    mo_coeff : np.ndarray
        Molecular orbital coefficients.
    mo_occ : np.ndarray
        Molecular orbital occupations.
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
        mo_coeff,
        mo_occ,
        compression="ia",
        compression_tol=1e-10,
        store_full=False,
    ):
        self.verbose = with_df.verbose
        self.stdout = with_df.stdout

        self.with_df = with_df
        self.mo_coeff = mo_coeff
        self.mo_occ = mo_occ
        self.compression = compression
        self.compression_tol = compression_tol
        self.store_full = store_full

        self._blocks = {}
        self._mo_coeff_g = None
        self._mo_coeff_w = None
        self._mo_occ_w = None
        self._rot = None

    def _parse_compression(self):
        """Parse the compression string."""

        if not self.compression:
            return set()

        compression = self.compression.replace("vo", "ov")
        compression = set(x for x in compression.split(","))

        if "ia" in compression and "ov" in compression:
            raise ValueError("`compression` cannot contain both `'ia'` and `'ov'` (or `'vo'`)")

        return compression

    def get_compression_metric(self):
        """
        Return the compression metric.

        Returns
        -------
        rot : numpy.ndarray
            Rotation matrix into the compressed auxiliary space.
        """

        compression = self._parse_compression()
        if not compression:
            return None

        cput0 = (logger.process_clock(), logger.perf_counter())
        logger.info(self, f"Computing compression metric for {self.__class__.__name__}")

        prod = np.zeros((self.naux_full, self.naux_full))

        # Loop over required blocks
        for key in sorted(compression):
            logger.debug(self, f"Transforming {key} block")
            ci, cj = [
                {
                    "o": self.mo_coeff[:, self.mo_occ > 0],
                    "v": self.mo_coeff[:, self.mo_occ == 0],
                    "i": self.mo_coeff_w[:, self.mo_occ_w > 0],
                    "a": self.mo_coeff_w[:, self.mo_occ_w == 0],
                }[k]
                for k in key
            ]
            ni, nj = ci.shape[-1], cj.shape[-1]

            for p0, p1 in mpi_helper.prange(0, ni * nj, self.with_df.blockdim):
                i0, j0 = divmod(p0, nj)
                i1, j1 = divmod(p1, nj)

                Lxy = np.zeros((self.naux_full, p1 - p0))
                b1 = 0
                for block in self.with_df.loop():
                    block = lib.unpack_tril(block)
                    b0, b1 = b1, b1 + block.shape[0]
                    logger.debug(self, f"  Block [{p0}:{p1}, {b0}:{b1}]")

                    tmp = lib.einsum("Lpq,pi,qj->Lij", block, ci[:, i0 : i1 + 1], cj)
                    tmp = tmp.reshape(b1 - b0, -1)
                    Lxy[b0:b1] = tmp[:, j0 : j0 + (p1 - p0)]

                prod += np.dot(Lxy, Lxy.T)

        prod = mpi_helper.allreduce(prod, root=0)

        if mpi_helper.rank == 0:
            e, v = np.linalg.eigh(prod)
            mask = np.abs(e) > self.compression_tol
            rot = v[:, mask]
        else:
            rot = np.zeros((0,))
        del prod

        rot = mpi_helper.bcast(rot, root=0)

        if rot.shape[-1] == self.naux_full:
            logger.info(self, "No compression found")
            rot = None
        else:
            logger.info(self, f"Compressed auxiliary space from {self.naux_full} to {rot.shape[1]}")
        logger.timer(self, "compression metric", *cput0)

        return rot

    def transform(self, do_Lpq=None, do_Lpx=True, do_Lia=True):
        """
        Transform the integrals.

        Parameters
        ----------
        do_Lpq : bool, optional
            Whether to compute the full (aux, MO, MO) array. Default
            value is `True` if `store_full` is `True`, `False`
            otherwise.
        do_Lpx : bool, optional
            Whether to compute the compressed (aux, MO, MO) array.
            Default value is `True`.
        do_Lia : bool, optional
            Whether to compute the compressed (aux, occ, vir) array.
            Default value is `True`.
        """

        # Get the compression metric
        if self._rot is None:
            self._rot = self.get_compression_metric()
        rot = self._rot
        if rot is None:
            rot = np.eye(self.naux_full)

        do_Lpq = self.store_full if do_Lpq is None else do_Lpq
        if not any([do_Lpq, do_Lpx, do_Lia]):
            return

        cput0 = (logger.process_clock(), logger.perf_counter())
        logger.info(self, f"Transforming {self.__class__.__name__}")

        # Get the slices on the current process and initialise the arrays
        o0, o1 = list(mpi_helper.prange(0, self.nmo, self.nmo))[0]
        p0, p1 = list(mpi_helper.prange(0, self.nmo_g, self.nmo_g))[0]
        q0, q1 = list(mpi_helper.prange(0, self.nocc_w * self.nvir_w, self.nocc_w * self.nvir_w))[0]
        Lpq = np.zeros((self.naux_full, self.nmo, o1 - o0)) if do_Lpq else None
        Lpx = np.zeros((self.naux, self.nmo, p1 - p0)) if do_Lpx else None
        Lia = np.zeros((self.naux, q1 - q0)) if do_Lia else None

        # Build the integrals blockwise
        b1 = 0
        for block in self.with_df.loop():
            block = lib.unpack_tril(block)
            b0, b1 = b1, b1 + block.shape[0]
            logger.debug(self, f"  Block [{b0}:{b1}]")

            # If needed, rotate the full (L|pq) array
            if do_Lpq:
                logger.debug(self, f"(L|pq) size: ({self.naux_full}, {self.nmo}, {o1 - o0})")
                coeffs = (self.mo_coeff, self.mo_coeff[:, o0:o1])
                Lpq[b0:b1] = lib.einsum("Lpq,pi,qj->Lij", block, *coeffs)

            # Compress the block
            block = lib.einsum("L...,LQ->Q...", block, rot[b0:b1])

            # Build the compressed (L|px) array
            if do_Lpx:
                logger.debug(self, f"(L|px) size: ({self.naux}, {self.nmo}, {p1 - p0})")
                coeffs = (self.mo_coeff, self.mo_coeff_g[:, p0:p1])
                Lpx += lib.einsum("Lpq,pi,qj->Lij", block, *coeffs)

            # Build the compressed (L|ia) array
            if do_Lia:
                logger.debug(self, f"(L|ia) size: ({self.naux}, {q1 - q0})")
                i0, a0 = divmod(q0, self.nvir_w)
                i1, a1 = divmod(q1, self.nvir_w)
                coeffs = (self.mo_coeff_w[:, i0 : i1 + 1], self.mo_coeff_w[:, self.nocc_w :])
                tmp = lib.einsum("Lpq,pi,qj->Lij", block, *coeffs)
                tmp = tmp.reshape(self.naux, -1)
                Lia += tmp[:, a0 : a0 + (q1 - q0)]

        if do_Lpq:
            self._blocks["Lpq"] = Lpq
        if do_Lpx:
            self._blocks["Lpx"] = Lpx
        if do_Lia:
            self._blocks["Lia"] = Lia

        logger.timer(self, "transform", *cput0)

    def update_coeffs(self, mo_coeff_g=None, mo_coeff_w=None, mo_occ_w=None):
        """
        Update the MO coefficients for the Green's function and the
        screened Coulomb interaction.

        Parameters
        ----------
        mo_coeff_g : numpy.ndarray, optional
            Coefficients corresponding to the Green's function. Default
            value is `None`.
        mo_coeff_w : numpy.ndarray, optional
            Coefficients corresponding to the screened Coulomb
            interaction. Default value is `None`.
        mo_occ_w : numpy.ndarray, optional
            Occupations corresponding to the screened Coulomb
            interaction. Default value is `None`.

        Notes
        -----
        If `mo_coeff_g` is `None`, the Green's function is assumed to
        remain in the basis in which it was originally defined, and
        vice-versa for `mo_coeff_w` and `mo_occ_w`. At least one of
        `mo_coeff_g` and `mo_coeff_w` must be provided.
        """

        if any((mo_coeff_w is not None, mo_occ_w is not None)):
            assert mo_coeff_w is not None and mo_occ_w is not None

        if mo_coeff_g is not None:
            self._mo_coeff_g = mo_coeff_g

        if mo_coeff_w is not None:
            self._mo_coeff_w = mo_coeff_w
            self._mo_occ_w = mo_occ_w
            if "ia" in self._parse_compression():
                self.rot = self.get_compression_metric()

        self.transform(
            do_Lpq=False,
            do_Lpx=mo_coeff_g is not None,
            do_Lia=mo_coeff_w is not None,
        )

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

        p0, p1 = list(mpi_helper.prange(0, self.nmo, self.nmo))[0]
        vj = np.zeros_like(dm, dtype=np.result_type(dm, self.dtype))

        if self.store_full and basis == "mo":
            tmp = lib.einsum("Qkl,lk->Q", self.Lpq, dm[p0:p1])
            tmp = mpi_helper.allreduce(tmp)
            vj[:, p0:p1] = lib.einsum("Qij,Q->ij", self.Lpq, tmp)
            vj = mpi_helper.allreduce(vj)

        else:
            if basis == "mo":
                dm = np.linalg.multi_dot((self.mo_coeff, dm, self.mo_coeff.T))

            with patch_df_loop(self.with_df):
                for block in self.with_df.loop():
                    naux = block.shape[0]
                    if block.size == naux * self.nmo * (self.nmo + 1) // 2:
                        block = lib.unpack_tril(block)
                    block = block.reshape(naux, self.nmo, self.nmo)

                    tmp = lib.einsum("Qkl,lk->Q", block, dm)
                    vj += lib.einsum("Qij,Q->ij", block, tmp)

            vj = mpi_helper.allreduce(vj)
            if basis == "mo":
                vj = np.linalg.multi_dot((self.mo_coeff.T, vj, self.mo_coeff))

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

        p0, p1 = list(mpi_helper.prange(0, self.nmo, self.nmo))[0]
        vk = np.zeros_like(dm, dtype=np.result_type(dm, self.dtype))

        if self.store_full and basis == "mo":
            tmp = lib.einsum("Qik,kl->Qil", self.Lpq, dm[p0:p1])
            tmp = mpi_helper.allreduce(tmp)
            vk[:, p0:p1] = lib.einsum("Qil,Qlj->ij", tmp, self.Lpq)
            vk = mpi_helper.allreduce(vk)

        else:
            if basis == "mo":
                dm = np.linalg.multi_dot((self.mo_coeff, dm, self.mo_coeff.T))

            with patch_df_loop(self.with_df):
                for block in self.with_df.loop():
                    naux = block.shape[0]
                    if block.size == naux * self.nmo * (self.nmo + 1) // 2:
                        block = lib.unpack_tril(block)
                    block = block.reshape(naux, self.nmo, self.nmo)

                    tmp = lib.einsum("Qik,kl->Qil", block, dm)
                    vk += lib.einsum("Qil,Qlj->ij", tmp, block)

            vk = mpi_helper.allreduce(vk)
            if basis == "mo":
                vk = np.linalg.multi_dot((self.mo_coeff.T, vk, self.mo_coeff))

        return vk

    def get_jk(self, dm, **kwargs):
        """Build the J and K matrices.

        Returns
        -------
        vj : numpy.ndarray
            J matrix.
        vk : numpy.ndarray
            K matrix.

        Notes
        -----
        See `get_j` and `get_k` for more information.
        """
        return self.get_j(dm, **kwargs), self.get_k(dm, **kwargs)

    def get_fock(self, dm, h1e, **kwargs):
        """Build the Fock matrix.

        Parameters
        ----------
        dm : numpy.ndarray
            Density matrix.
        h1e : numpy.ndarray
            Core Hamiltonian matrix.
        **kwargs : dict, optional
            Additional keyword arguments for `get_jk`.

        Returns
        -------
        fock : numpy.ndarray
            Fock matrix.

        Notes
        -----
        See `get_jk` for more information. The basis of `h1e` must be
        the same as `dm`.
        """
        vj, vk = self.get_jk(dm, **kwargs)
        return h1e + vj - vk * 0.5

    @property
    def Lpq(self):
        """Return the full uncompressed (aux, MO, MO) array."""
        return self._blocks["Lpq"]

    @property
    def Lpx(self):
        """Return the compressed (aux, MO, G) array."""
        return self._blocks["Lpx"]

    @property
    def Lia(self):
        """Return the compressed (aux, W occ, W vir) array."""
        return self._blocks["Lia"]

    @property
    def mo_coeff_g(self):
        """Return the MO coefficients for the Green's function."""
        return self._mo_coeff_g if self._mo_coeff_g is not None else self.mo_coeff

    @property
    def mo_coeff_w(self):
        """
        Return the MO coefficients for the screened Coulomb interaction.
        """
        return self._mo_coeff_w if self._mo_coeff_w is not None else self.mo_coeff

    @property
    def mo_occ_w(self):
        """
        Return the MO occupation numbers for the screened Coulomb interaction.
        """
        return self._mo_occ_w if self._mo_occ_w is not None else self.mo_occ

    @property
    def nmo(self):
        """Return the number of MOs."""
        return self.mo_coeff.shape[-1]

    @property
    def nocc(self):
        """Return the number of occupied MOs."""
        return np.sum(self.mo_occ > 0)

    @property
    def nvir(self):
        """Return the number of virtual MOs."""
        return np.sum(self.mo_occ == 0)

    @property
    def nmo_g(self):
        """Return the number of MOs for the Green's function."""
        return self.mo_coeff_g.shape[-1]

    @property
    def nmo_w(self):
        """
        Return the number of MOs for the screened Coulomb interaction.
        """
        return self.mo_coeff_w.shape[-1]

    @property
    def nocc_w(self):
        """
        Return the number of occupied MOs for the screened Coulomb
        interaction.
        """
        return np.sum(self.mo_occ_w > 0)

    @property
    def nvir_w(self):
        """
        Return the number of virtual MOs for the screened Coulomb
        interaction.
        """
        return np.sum(self.mo_occ_w == 0)

    @property
    def naux(self):
        """
        Return the number of auxiliary basis functions, after the
        compression.
        """
        if self._rot is None:
            return self.naux_full
        return self._rot.shape[1]

    @property
    def naux_full(self):
        """
        Return the number of auxiliary basis functions, before the
        compression.
        """
        return self.with_df.get_naoaux()

    @property
    def is_bare(self):
        """
        Return a boolean flag indicating whether the integrals have
        no self-consistencies.
        """
        return self._mo_coeff_g is None and self._mo_coeff_w is None

    @property
    def dtype(self):
        """
        Return the dtype of the integrals.
        """
        return np.result_type(*self._blocks.values())
