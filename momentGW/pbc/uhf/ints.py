"""
Integral helpers with periodic boundary conditions and unrestricted
reference.
"""

import numpy as np
from pyscf import lib
from pyscf.agf2 import mpi_helper
from pyscf.lib import logger

from momentGW.pbc.ints import KIntegrals
from momentGW.uhf.ints import UIntegrals


class KIntegrals_α(KIntegrals):
    """Overload the `__name__` to signify α part"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__class__.__name__ = "KIntegrals (α)"

    def get_compression_metric(self):
        return None


class KIntegrals_β(KIntegrals):
    """Overload the `__name__` to signify β part"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__class__.__name__ = "KIntegrals (β)"

    def get_compression_metric(self):
        return None


class KUIntegrals(UIntegrals, KIntegrals):
    """
    Container for the integrals required for KUGW methods.

    Parameters
    ----------
    with_df : pyscf.pbc.df.DF
        Density fitting object.
    mo_coeff : np.ndarray
        Molecular orbital coefficients for each k-point for each spin
        channel.
    mo_occ : np.ndarray
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
        self.verbose = with_df.verbose
        self.stdout = with_df.stdout

        self.with_df = with_df
        self.kpts = kpts
        self.mo_coeff = mo_coeff
        self.mo_occ = mo_occ
        self.compression = compression
        self.compression_tol = compression_tol
        self.store_full = store_full

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

        cput0 = (logger.process_clock(), logger.perf_counter())
        logger.info(self, f"Computing compression metric for {self.__class__.__name__}")

        prod = np.zeros((len(self.kpts), self.naux_full, self.naux_full), dtype=complex)

        # Loop over required blocks
        for key in sorted(compression):
            for s, spin in enumerate(["α", "β"]):
                logger.debug(self, f"Transforming {key} block ({spin})")
                ci, cj = [
                    {
                        "o": [c[:, o > 0] for c, o in zip(self.mo_coeff[s], self.mo_occ[s])],
                        "v": [c[:, o == 0] for c, o in zip(self.mo_coeff[s], self.mo_occ[s])],
                        "i": [c[:, o > 0] for c, o in zip(self.mo_coeff_w[s], self.mo_occ_w[s])],
                        "a": [c[:, o == 0] for c, o in zip(self.mo_coeff_w[s], self.mo_occ_w[s])],
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
                        logger.debug(self, f"  Block [{ki}, {kj}, {b0}:{b1}]")

                        tmp = lib.einsum("Lpq,pi,qj->Lij", block, ci[ki].conj(), cj[kj])
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

        for q in self.kpts.loop(1):
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
