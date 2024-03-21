"""
Integral helpers with unrestricted reference.
"""

import numpy as np
from pyscf.ao2mo import _ao2mo

from momentGW import logging, mpi_helper
from momentGW.ints import Integrals


class Integrals_α(Integrals):
    """Extends `Integrals` to represent the α channel."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__class__.__name__ = "Integrals (α)"

    def get_compression_metric(self):
        """Return the compression metric.

        Overrides `Integrals.get_compression_metric` to return `None`,
        as the compression metric should be calculated for spinless
        auxiliaries.

        Returns
        -------
        rot : numpy.ndarray
            Rotation matrix into the compressed auxiliary space.
        """
        return None


class Integrals_β(Integrals_α):
    """Overload the `__name__` to signify β part"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__class__.__name__ = "Integrals (β)"


class UIntegrals(Integrals):
    """
    Container for the density-fitted integrals required for UGW methods.

    Parameters
    ----------
    with_df : pyscf.df.DF
        Density fitting object.
    mo_coeff : numpy.ndarray
        Molecular orbital coefficients for each spin channel.
    mo_occ : numpy.ndarray
        Molecular orbital occupations for each spin channel.
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
        # Parameters
        self.with_df = with_df
        self.mo_coeff = mo_coeff
        self.mo_occ = mo_occ

        # Options
        self.compression = compression
        self.compression_tol = compression_tol
        self.store_full = store_full

        # Attributes
        self._spins = {
            0: Integrals_α(
                self.with_df,
                self.mo_coeff[0],
                self.mo_occ[0],
                compression=self.compression,
                compression_tol=self.compression_tol,
                store_full=self.store_full,
            ),
            1: Integrals_β(
                self.with_df,
                self.mo_coeff[1],
                self.mo_occ[1],
                compression=self.compression,
                compression_tol=self.compression_tol,
                store_full=self.store_full,
            ),
        }

    @logging.with_status("Computing compression metric")
    def get_compression_metric(self):
        """
        Return the compression metric.

        Returns
        -------
        rot : numpy.ndarray
            Rotation matrix into the compressed auxiliary space.
        """

        # Get the compression sectors
        compression = self._parse_compression()
        if not compression:
            return None

        # Initialise the inner product matrix
        prod = np.zeros((self.naux_full, self.naux_full))

        # Loop over required blocks
        for key in sorted(compression):
            for s, spin in enumerate(["α", "β"]):
                with logging.with_status(f"{key} ({spin}) sector"):
                    # Get the coefficients
                    ci, cj = [
                        {
                            "o": self.mo_coeff[s][:, self.mo_occ[s] > 0],
                            "v": self.mo_coeff[s][:, self.mo_occ[s] == 0],
                            "i": self.mo_coeff_w[s][:, self.mo_occ_w[s] > 0],
                            "a": self.mo_coeff_w[s][:, self.mo_occ_w[s] == 0],
                        }[k]
                        for k in key
                    ]
                    ni, nj = ci.shape[-1], cj.shape[-1]
                    coeffs = np.concatenate((ci, cj), axis=1)

                    # Loop over the blocks
                    for p0, p1 in mpi_helper.prange(0, ni * nj, self.with_df.blockdim):
                        i0, j0 = divmod(p0, nj)
                        i1, j1 = divmod(p1, nj)

                        # Build the (L|xy) array
                        Lxy = np.zeros((self.naux_full, p1 - p0))
                        b1 = 0
                        for block in self.with_df.loop():
                            b0, b1 = b1, b1 + block.shape[0]
                            progress = (p0 * self.naux_full + b0) / (ni * nj * self.naux_full)
                            with logging.with_status(
                                f"block [{p0}:{p1}, {b0}:{b1}] ({progress:.1%})"
                            ):
                                tmp = _ao2mo.nr_e2(
                                    block,
                                    coeffs,
                                    (i0, i1 + 1, ni, ni + nj),
                                    aosym="s2",
                                    mosym="s1",
                                )
                                tmp = tmp.reshape(b1 - b0, -1)
                                Lxy[b0:b1] = tmp[:, j0 : j0 + (p1 - p0)]

                        # Update the inner product matrix
                        prod += np.dot(Lxy, Lxy.T)

        # Reduce the inner product matrix
        prod = mpi_helper.allreduce(prod, root=0)
        prod *= 0.5

        # Diagonalise the inner product matrix
        if mpi_helper.rank == 0:
            e, v = np.linalg.eigh(prod)
            mask = np.abs(e) > self.compression_tol
            rot = v[:, mask]
        else:
            rot = np.zeros((0,))
        del prod

        # Broadcast the rotation matrix in case of hybrid parallelism
        # introducing non-determinism
        rot = mpi_helper.bcast(rot, root=0)

        # Print the compression status
        if rot.shape[-1] == self.naux_full:
            logging.write("No compression found for auxiliary space")
            rot = None
        else:
            percent = 100 * rot.shape[-1] / self.naux_full
            style = logging.rate(percent, 80, 95)
            logging.write(
                f"Compressed auxiliary space from {self.naux_full} to {rot.shape[1]} "
                f"([{style}]{percent:.1f}%)[/]"
            )

        return rot

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
        if self._spins[0]._rot is None:
            self._spins[0]._rot = self._spins[1]._rot = self.get_compression_metric()

        # Transform the integrals
        self._spins[0].transform(do_Lpq=do_Lpq, do_Lpx=do_Lpx, do_Lia=do_Lia)
        self._spins[1].transform(do_Lpq=do_Lpq, do_Lpx=do_Lpx, do_Lia=do_Lia)

    def update_coeffs(self, mo_coeff_g=None, mo_coeff_w=None, mo_occ_w=None):
        """
        Update the MO coefficients for the Green's function and the
        screened Coulomb interaction.

        Parameters
        ----------
        mo_coeff_g : numpy.ndarray, optional
            Coefficients corresponding to the Green's function for each
            spin. Default value is `None`.
        mo_coeff_w : numpy.ndarray, optional
            Coefficients corresponding to the screened Coulomb
            interaction for each spin. Default value is `None`.
        mo_occ_w : numpy.ndarray, optional
            Occupations corresponding to the screened Coulomb
            interaction for each spin. Default value is `None`.

        Notes
        -----
        If `mo_coeff_g` is `None`, the Green's function is assumed to
        remain in the basis in which it was originally defined, and
        vice-versa for `mo_coeff_w` and `mo_occ_w`. At least one of
        `mo_coeff_g` and `mo_coeff_w` must be provided.
        """

        # Check the input
        if any((mo_coeff_w is not None, mo_occ_w is not None)):
            assert mo_coeff_w is not None and mo_occ_w is not None

        # Update the Green's function coefficients
        if mo_coeff_g is not None:
            self._spins[0]._mo_coeff_g = mo_coeff_g[0]
            self._spins[1]._mo_coeff_g = mo_coeff_g[1]

        # Update the screened Coulomb interaction coefficients
        do_all = False
        rot = None
        if mo_coeff_w is not None:
            self._spins[0]._mo_coeff_w = mo_coeff_w[0]
            self._spins[1]._mo_coeff_w = mo_coeff_w[1]
            self._spins[0]._mo_occ_w = mo_occ_w[0]
            self._spins[1]._mo_occ_w = mo_occ_w[1]
            if "ia" in self._parse_compression():
                do_all = True
                rot = self.get_compression_metric()

        # Set the compression metric
        self._spins[0]._rot = rot
        self._spins[1]._rot = rot

        # Transform the integrals
        self.transform(
            do_Lpq=self.store_full and do_all,
            do_Lpx=mo_coeff_g is not None or do_all,
            do_Lia=mo_coeff_w is not None or do_all,
        )

    def get_j(self, dm, basis="mo"):
        """Build the J matrix.

        Parameters
        ----------
        dm : numpy.ndarray
            Density matrix for each spin channel.
        basis : str, optional
            Basis in which to build the J matrix. One of
            `("ao", "mo")`. Default value is `"mo"`.

        Returns
        -------
        vj : tuple of numpy.ndarray
            J matrix for each spin channel.
        """

        # Calculate Coulomb term each pair of spins
        vj_αα = self._spins[0].get_j(dm[0], basis=basis, other=self._spins[0])
        vj_αβ = self._spins[0].get_j(dm[1], basis=basis, other=self._spins[1])
        vj_ββ = self._spins[1].get_j(dm[1], basis=basis, other=self._spins[1])
        vj_βα = self._spins[1].get_j(dm[0], basis=basis, other=self._spins[0])

        # Build the J matrix for each spin
        vj = np.array([vj_αα + vj_αβ, vj_ββ + vj_βα])

        return vj

    def get_k(self, dm, basis="mo"):
        """Build the K matrix.

        Parameters
        ----------
        dm : numpy.ndarray
            Density matrix for each spin channel.
        basis : str, optional
            Basis in which to build the K matrix. One of
            `("ao", "mo")`. Default value is `"mo"`.

        Returns
        -------
        vk : numpy.ndarray
            K matrix for each spin channel.
        """

        # Calculate exchange term each spin
        vk_αα = self._spins[0].get_k(dm[0], basis=basis)
        vk_ββ = self._spins[1].get_k(dm[1], basis=basis)
        vk = np.array([vk_αα, vk_ββ])

        return vk

    def get_jk(self, dm, **kwargs):
        """Build the J and K matrices.

        Returns
        -------
        vj : numpy.ndarray
            J matrix for each spin channel.
        vk : numpy.ndarray
            K matrix for each spin channel.

        Notes
        -----
        See `get_j` and `get_k` for more information.
        """
        return super().get_jk(dm, **kwargs)

    def get_veff(self, dm, j=None, k=None, **kwargs):
        """Build the effective potential.

        Parameters
        ----------
        dm : numpy.ndarray
            Density matrix for each spin channel.
        j : numpy.ndarray, optional
            J matrix for each spin channel. If `None`, compute it.
            Default value is `None`.
        k : numpy.ndarray, optional
            K matrix for each spin channel. If `None`, compute it.
            Default value is `None`.
        **kwargs : dict, optional
            Additional keyword arguments for `get_jk`.

        Returns
        -------
        veff : numpy.ndarray
            Effective potential.

        Notes
        -----
        See `get_jk` for more information.
        """
        if j is None and k is None:
            vj, vk = self.get_jk(dm, **kwargs)
        elif j is None:
            vj, vk = self.get_j(dm, **kwargs), k
        elif k is None:
            vj, vk = j, self.get_k(dm, **kwargs)
        return vj - vk

    def get_fock(self, dm, h1e, **kwargs):
        """Build the Fock matrix.

        Parameters
        ----------
        dm : numpy.ndarray
            Density matrix for each spin channel.
        h1e : numpy.ndarray
            Core Hamiltonian matrix for each spin channel.
        **kwargs : dict, optional
            Additional keyword arguments for `get_jk`.

        Returns
        -------
        fock : numpy.ndarray
            Fock matrix for each spin channel.

        Notes
        -----
        See `get_jk` for more information. The basis of `h1e` must be
        the same as `dm`.
        """
        return super().get_fock(dm, **kwargs)

    @property
    def Lpq(self):
        """Get the full uncompressed ``(aux, MO, MO)`` integrals."""
        return (self._spins[0].Lpq, self._spins[1].Lpq)

    @property
    def Lpx(self):
        """Get the compressed ``(aux, MO, MO)`` integrals."""
        return (self._spins[0].Lpx, self._spins[1].Lpx)

    @property
    def Lia(self):
        """Get the compressed ``(aux, occ, vir)`` integrals."""
        return (self._spins[0].Lia, self._spins[1].Lia)

    @property
    def mo_coeff_g(self):
        """Get the MO coefficients for the Green's function."""
        return (self._spins[0].mo_coeff_g, self._spins[1].mo_coeff_g)

    @property
    def mo_coeff_w(self):
        """Get the MO coefficients for the screened Coulomb interaction."""
        return (self._spins[0].mo_coeff_w, self._spins[1].mo_coeff_w)

    @property
    def mo_occ_w(self):
        """
        Get the MO occupation numbers for the screened Coulomb
        interaction.
        """
        return (self._spins[0].mo_occ_w, self._spins[1].mo_occ_w)

    @property
    def nmo(self):
        """Get the number of MOs."""
        return (self._spins[0].nmo, self._spins[1].nmo)

    @property
    def nocc(self):
        """Get the number of occupied MOs."""
        return (self._spins[0].nocc, self._spins[1].nocc)

    @property
    def nvir(self):
        """Get the number of virtual MOs."""
        return (self._spins[0].nvir, self._spins[1].nvir)

    @property
    def nmo_g(self):
        """Get the number of MOs for the Green's function."""
        return (self._spins[0].nmo_g, self._spins[1].nmo_g)

    @property
    def nmo_w(self):
        """Get the number of MOs for the screened Coulomb interaction."""
        return (self._spins[0].nmo_w, self._spins[1].nmo_w)

    @property
    def nocc_w(self):
        """
        Get the number of occupied MOs for the screened Coulomb
        interaction.
        """
        return (self._spins[0].nocc_w, self._spins[1].nocc_w)

    @property
    def nvir_w(self):
        """
        Get the number of virtual MOs for the screened Coulomb
        interaction.
        """
        return (self._spins[0].nvir_w, self._spins[1].nvir_w)

    @property
    def naux(self):
        """
        Get the number of auxiliary basis functions, after the
        compression.
        """
        assert self._spins[0].naux == self._spins[1].naux
        return self._spins[0].naux

    @property
    def naux_full(self):
        """
        Get the number of auxiliary basis functions, before the
        compression.
        """
        return self.with_df.get_naoaux()

    @property
    def is_bare(self):
        """
        Get a boolean flag indicating whether the integrals have
        no self-consistencies.
        """
        return self._mo_coeff_g is None and self._mo_coeff_w is None

    @property
    def dtype(self):
        """Get the dtype of the integrals."""
        return np.result_type(self._spins[0].dtype, self._spins[1].dtype)

    def __getitem__(self, key):
        """Get the integrals for one spin."""
        return self._spins[key]
