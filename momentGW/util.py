"""Utility functions.
"""

import numpy as np
import scipy.linalg
from pyscf import lib


class DIIS(lib.diis.DIIS):
    """
    Direct inversion of the iterative subspace (DIIS).

    For some reason, the default pyscf DIIS object can result in fully
    linearly dependent error vectors in high-moment self-consistent
    calculations. This class is a drop-in replacement with a fallback
    in this case.
    """

    def update_with_scaling(self, x, axis, xerr=None):
        """Scales the arrays, according to the maximum absolute value
        along given axis, executes DIIS, and then rescales the output.
        """

        scale = np.max(np.abs(x), axis=axis, keepdims=True)

        x = x / scale
        if xerr:
            xerr = xerr / scale
        x = self.update(x, xerr=xerr)
        x = x * scale

        return x

    def extrapolate(self, nd=None):
        if nd is None:
            nd = self.get_num_vec()
        if nd == 0:
            raise RuntimeError("No vector found in DIIS object.")

        h = self._H[: nd + 1, : nd + 1]
        g = np.zeros(nd + 1, h.dtype)
        g[0] = 1

        w, v = scipy.linalg.eigh(h)
        if np.any(abs(w) < 1e-14):
            lib.logger.debug(self, "Linear dependence found in DIIS error vectors.")
            idx = abs(w) > 1e-14
            c = np.dot(v[:, idx] * (1.0 / w[idx]), np.dot(v[:, idx].T.conj(), g))
        else:
            try:
                c = np.linalg.solve(h, g)
            except np.linalg.linalg.LinAlgError as e:
                lib.logger.warn(self, " diis singular, eigh(h) %s", w)
                raise e
        lib.logger.debug1(self, "diis-c %s", c)

        if np.all(abs(c) < 1e-14):
            raise np.linalg.linalg.LinAlgError("DIIS vectors are fully linearly dependent.")

        xnew = None
        for i, ci in enumerate(c[1:]):
            xi = self.get_vec(i)
            if xnew is None:
                xnew = np.zeros(xi.size, c.dtype)
            for p0, p1 in lib.prange(0, xi.size, lib.diis.BLOCK_SIZE):
                xnew[p0:p1] += xi[p0:p1] * ci
        return xnew


class SilentSCF:
    """
    Context manager to shut PySCF's SCF classes up.
    """

    def __init__(self, mf):
        self.mf = mf

    def __enter__(self):
        self._mol_verbose = self.mf.mol.verbose
        self.mf.mol.verbose = 0

        self._mf_verbose = self.mf.verbose
        self.mf.verbose = 0

        if getattr(self.mf, "with_df", None):
            self._df_verbose = self.mf.with_df.verbose
            self.mf.with_df.verbose = 0

        return self.mf

    def __exit__(self, exc_type, exc_value, traceback):
        self.mf.mol.verbose = self._mol_verbose
        self.mf.verbose = self._mf_verbose
        if getattr(self.mf, "with_df", None):
            self.mf.with_df.verbose = self._df_verbose
