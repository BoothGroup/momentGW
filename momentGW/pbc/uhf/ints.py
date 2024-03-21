"""
Integral helpers with periodic boundary conditions and unrestricted
reference.
"""

import numpy as np

from momentGW import logging, mpi_helper, util
from momentGW.pbc.ints import KIntegrals
from momentGW.uhf.ints import UIntegrals


class KIntegrals_α(KIntegrals):
    """Overload the `__name__` to signify α part"""

    def get_compression_metric(self):  # noqa: D102
        return None

    get_compression_metric.__doc__ = KIntegrals.get_compression_metric.__doc__

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__class__.__name__ = "KIntegrals (α)"


class KIntegrals_β(KIntegrals):
    """Overload the `__name__` to signify β part"""

    def get_compression_metric(self):  # noqa: D102
        return None

    get_compression_metric.__doc__ = KIntegrals.get_compression_metric.__doc__

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__class__.__name__ = "KIntegrals (β)"


class KUIntegrals(UIntegrals, KIntegrals):
    """
    Container for the integrals required for KUGW methods.

    Parameters
    ----------
    with_df : pyscf.pbc.df.DF
        Density fitting object.
    mo_coeff : numpy.ndarray
        Molecular orbital coefficients for each k-point for each spin
        channel.
    mo_occ : numpy.ndarray
        Molecular orbital occupations for each k-point for each spin
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
        self.with_df = with_df
        self.mo_coeff = mo_coeff
        self.mo_occ = mo_occ

        # Options
        self.compression = compression
        self.compression_tol = compression_tol
        self.store_full = store_full

        # Attributes
        self.kpts = kpts
        self._spins = {
            0: KIntegrals_α(
                self.with_df,
                self.kpts,
                self.mo_coeff[0],
                self.mo_occ[0],
                compression=self.compression,
                compression_tol=self.compression_tol,
                store_full=self.store_full,
            ),
            1: KIntegrals_β(
                self.with_df,
                self.kpts,
                self.mo_coeff[1],
                self.mo_occ[1],
                compression=self.compression,
                compression_tol=self.compression_tol,
                store_full=self.store_full,
            ),
        }
        self._madelung = None

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

        compression = self._parse_compression()
        if not compression:
            return None

        prod = np.zeros((len(self.kpts), self.naux_full, self.naux_full), dtype=complex)

        # Loop over required blocks
        for key in sorted(compression):
            for s, spin in enumerate(["α", "β"]):
                with logging.with_status(f"{key} ({spin}) sector"):
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

                        Lxy = np.zeros((self.naux_full, ni[ki] * nj[kj]), dtype=complex)
                        b1 = 0
                        for block in self.with_df.sr_loop((ki, kj), compact=False):
                            if block[2] == -1:
                                raise NotImplementedError("Low dimensional integrals")
                            block = block[0] + block[1] * 1.0j
                            block = block.reshape(self.naux_full, self.nmo[s], self.nmo[s])
                            b0, b1 = b1, b1 + block.shape[0]
                            progress = ki * len(self.kpts) ** 2 + kj * len(self.kpts) + b0
                            progress /= len(self.kpts) ** 2 + self.naux_full

                            with logging.with_status(
                                f"block [{ki}, {kj}, {b0}:{b1}] ({progress:.1%})"
                            ):
                                # TODO optimise
                                tmp = util.einsum("Lpq,pi,qj->Lij", block, ci[ki].conj(), cj[kj])
                                tmp = tmp.reshape(b1 - b0, -1)
                                Lxy[b0:b1] = tmp

                        prod[q] += np.dot(Lxy, Lxy.T.conj()) / len(self.kpts)

        prod *= 0.5

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

        naux_total = sum(r.shape[-1] for r in rot)
        naux_full_total = self.naux_full * len(self.kpts)
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
