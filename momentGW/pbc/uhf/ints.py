"""Integral helpers with periodic boundary conditions and unrestricted
reference.
"""

import numpy as np

from momentGW import logging, mpi_helper, util
from momentGW.pbc.ints import KIntegrals
from momentGW.uhf.ints import UIntegrals


class _KIntegrals_α(KIntegrals):
    """Extends `KIntegrals` to represent the α channel."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__class__.__name__ = "KIntegrals (α)"

    def get_compression_metric(self):
        """Return the compression metric.

        Overrides `KIntegrals.get_compression_metric` to return `None`,
        as the compression metric should be calculated for spinless
        auxiliaries.

        Returns
        -------
        rot : numpy.ndarray
            Rotation matrix into the compressed auxiliary space.
        """
        return None


class _KIntegrals_β(_KIntegrals_α):
    """Extends `KIntegrals` to represent the β channel."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__class__.__name__ = "KIntegrals (β)"


class KUIntegrals(UIntegrals, KIntegrals):
    """Container for the density-fitted integrals required for KUGW
    methods.

    Parameters
    ----------
    with_df : pyscf.pbc.df.DF
        Density fitting object.
    mo_coeff : numpy.ndarray
        Molecular orbital coefficients at each k-point for each spin
        channel.
    mo_occ : numpy.ndarray
        Molecular orbital occupations at each k-point for each spin
        channel.
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
    ):
        # Parameters
        self._with_df = with_df
        self._mo_coeff = mo_coeff
        self._mo_occ = mo_occ

        # Options
        self.compression = compression
        self.compression_tol = compression_tol
        self.store_full = store_full

        # Attributes
        self.kpts = kpts
        self._spins = {
            0: _KIntegrals_α(
                self.with_df,
                self.kpts,
                self.mo_coeff[0],
                self.mo_occ[0],
                compression=self.compression,
                compression_tol=self.compression_tol,
                store_full=self.store_full,
            ),
            1: _KIntegrals_β(
                self.with_df,
                self.kpts,
                self.mo_coeff[1],
                self.mo_occ[1],
                compression=self.compression,
                compression_tol=self.compression_tol,
                store_full=self.store_full,
            ),
        }

    @logging.with_status("Computing compression metric")
    def get_compression_metric(self):
        """Return the compression metric.

        Returns
        -------
        rot : numpy.ndarray
            Rotation matrix into the compressed auxiliary space.
        """

        # Initialise the sizes
        naux_full = self.naux_full

        # Get the compression sectors
        compression = self._parse_compression()
        if not compression:
            return None

        # Initialise the inner product matrix
        prod = np.zeros((len(self.kpts), naux_full, naux_full), dtype=complex)

        # Loop over required blocks
        for key in sorted(compression):
            for s, spin in enumerate(["α", "β"]):
                with logging.with_status(f"{key} ({spin}) sector"):
                    # Get the coefficients
                    ci, cj = [
                        {
                            "o": [c[:, o > 0] for c, o in zip(self.mo_coeff[s], self.mo_occ[s])],
                            "v": [c[:, o == 0] for c, o in zip(self.mo_coeff[s], self.mo_occ[s])],
                            "i": [
                                c[:, o > 0] for c, o in zip(self.mo_coeff_w[s], self.mo_occ_w[s])
                            ],
                            "a": [
                                c[:, o == 0] for c, o in zip(self.mo_coeff_w[s], self.mo_occ_w[s])
                            ],
                        }[k]
                        for k in key
                    ]
                    ni = [c.shape[-1] for c in ci]
                    nj = [c.shape[-1] for c in cj]

                    for q, ki in self.kpts.loop(2):
                        kj = self.kpts.member(self.kpts.wrap_around(self.kpts[ki] - self.kpts[q]))

                        # Build the (L|xy) array
                        Lxy = np.zeros((naux_full, ni[ki] * nj[kj]), dtype=complex)
                        b1 = 0
                        for block in self.with_df.sr_loop((ki, kj), compact=False):
                            if block[2] == -1:
                                raise NotImplementedError("Low dimensional integrals")
                            block = block[0] + block[1] * 1.0j
                            block = block.reshape(naux_full, self.nao, self.nao)
                            b0, b1 = b1, b1 + block.shape[0]
                            progress = ki * len(self.kpts) ** 2 + kj * len(self.kpts) + b0
                            progress /= len(self.kpts) ** 2 + naux_full

                            with logging.with_status(
                                f"block [{ki}, {kj}, {b0}:{b1}] ({progress:.1%})"
                            ):
                                # TODO optimise
                                tmp = util.einsum("Lpq,pi,qj->Lij", block, ci[ki].conj(), cj[kj])
                                tmp = tmp.reshape(b1 - b0, -1)
                                Lxy[b0:b1] = tmp

                        # Update the inner product matrix
                        prod[q] += np.dot(Lxy, Lxy.T.conj()) / len(self.kpts)

        prod *= 0.5

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
        naux_full_total = naux_full * len(self.kpts)
        if naux_total == naux_full_total:
            logging.write("No compression found for auxiliary space")
            rot = None
        else:
            percent = 100 * naux_total / naux_full_total
            style = logging.rate(percent, 80, 95)
            logging.write(
                f"Compressed auxiliary space from {naux_full_total} to {naux_total} "
                f"([{style}]{percent:.1f}%)[/]"
            )

        return rot

    def update_coeffs(self, mo_coeff_g=None, mo_coeff_w=None, mo_occ_w=None):
        """Update the MO coefficients for the Green's function and the
        screened Coulomb interaction.

        Parameters
        ----------
        mo_coeff_g : numpy.ndarray, optional
            Coefficients corresponding to the Green's function at each
            k-point for each spin channel. Default value is `None`.
        mo_coeff_w : numpy.ndarray, optional
            Coefficients corresponding to the screened Coulomb
            interaction at each k-point for each spin channel. Default
            value is `None`.
        mo_occ_w : numpy.ndarray, optional
            Occupations corresponding to the screened Coulomb
            interaction at each k-point for each spin channel. Default
            value is `None`.

        Notes
        -----
        If `mo_coeff_g` is `None`, the Green's function is assumed to
        remain in the basis in which it was originally defined, and
        vice-versa for `mo_coeff_w` and `mo_occ_w`. At least one of
        `mo_coeff_g` and `mo_coeff_w` must be provided.
        """
        return super().update_coeffs(
            mo_coeff_g=mo_coeff_g,
            mo_coeff_w=mo_coeff_w,
            mo_occ_w=mo_occ_w,
        )

    def get_j(self, dm, basis="mo"):
        """Build the J matrix.

        Parameters
        ----------
        dm : numpy.ndarray
            Density matrix at each k-point for each spin channel.
        basis : str, optional
            Basis in which to build the J matrix. One of
            `("ao", "mo")`. Default value is `"mo"`.

        Returns
        -------
        vj : numpy.ndarray
            J matrix at each k-point for each spin channel.
        """
        return super().get_j(dm, basis=basis)

    def get_k(self, dm, basis="mo"):
        """Build the K matrix.

        Parameters
        ----------
        dm : numpy.ndarray
            Density matrix at each k-point for each spin channel.
        basis : str, optional
            Basis in which to build the K matrix. One of
            `("ao", "mo")`. Default value is `"mo"`.

        Returns
        -------
        vk : numpy.ndarray
            K matrix for each spin channel.
        """
        return super().get_k(dm, basis=basis)

    def get_jk(self, dm, **kwargs):
        """Build the J and K matrices.

        Returns
        -------
        vj : numpy.ndarray
            J matrix at each k-point for each spin channel.
        vk : numpy.ndarray
            K matrix at each k-point for each spin channel.

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
            Density matrix at each k-point for each spin channel.
        j : numpy.ndarray, optional
            J matrix at each k-point for each spin channel. If `None`,
            compute it. Default value is `None`.
        k : numpy.ndarray, optional
            K matrix at each k-point for each spin channel. If `None`,
            compute it. Default value is `None`.
        **kwargs : dict, optional
            Additional keyword arguments for `get_jk`.

        Returns
        -------
        veff : numpy.ndarray
            Effective potential at each k-point for each spin channel.

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
            Density matrix at each k-point for each spin channel.
        h1e : numpy.ndarray
            Core Hamiltonian matrix at each k-point for each spin
            channel.
        **kwargs : dict, optional
            Additional keyword arguments for `get_jk`.

        Returns
        -------
        fock : numpy.ndarray
            Fock matrix at each k-point for each spin channel.

        Notes
        -----
        See `get_jk` for more information. The basis of `h1e` must be
        the same as `dm`.
        """
        return super().get_fock(dm, h1e, **kwargs)
