"""Utility functions.
"""

import numpy as np
import scipy.linalg
from pyscf import lib


class DIIS(lib.diis.DIIS):
    """
    Direct inversion of the iterative subspace (DIIS).

    See `pyscf.lib.diis.DIIS` for more information.

    Notes
    -----
    For some reason, the default pyscf DIIS object can result in fully
    linearly dependent error vectors in high-moment self-consistent
    calculations. This class is a drop-in replacement with a fallback
    in this case.
    """

    def update_with_scaling(self, x, axis, xerr=None):
        """
        Scales the arrays, according to the maximum absolute value along
        given axis, executes DIIS, and then rescales the output.

        Parameters
        ----------
        x : numpy.ndarray
            Array to update with DIIS.
        axis : int or tuple
            Axis or axes along which to scale.
        xerr : numpy.ndarray, optional
            Error metric for the array. Default is `None`.

        Returns
        -------
        x : numpy.ndarray
            Updated array.

        Notes
        -----
        This function is useful for extrapolations on moments which span
        several orders of magnitude.
        """

        scale = np.max(np.abs(x), axis=axis, keepdims=True)

        # Scale
        x = x / scale
        if xerr:
            xerr = xerr / scale

        # Execute DIIS
        x = self.update(x, xerr=xerr)

        # Rescale
        x = x * scale

        return x

    def update_with_complex_unravel(self, x, xerr=None):
        """
        Execute DIIS where the error vectors are unravelled to
        concatenate the real and imaginary parts.

        Parameters
        ----------
        x : numpy.ndarray
            Array to update with DIIS.
        xerr : numpy.ndarray, optional
            Error metric for the array. Default is `None`.

        Returns
        -------
        x : numpy.ndarray
            Updated array.
        """

        if not np.iscomplexobj(x):
            return self.update(x, xerr=xerr)

        shape = x.shape
        size = x.size

        # Concatenate
        x = np.concatenate([np.real(x).ravel(), np.imag(x).ravel()])
        if xerr is not None:
            xerr = np.concatenate([np.real(xerr).ravel(), np.imag(xerr).ravel()])

        # Execute DIIS
        x = self.update(x, xerr=xerr)

        # Unravel
        x = x[:size] + 1j * x[size:]
        x = x.reshape(shape)

        return x

    def extrapolate(self, nd=None):
        """
        Extrapolate the DIIS vectors.

        Parameters
        ----------
        nd : int, optional
            Number of vectors to extrapolate. Default is `None`, which
            extrapolates all vectors.

        Returns
        -------
        xnew : numpy.ndarray
            Extrapolated vector.

        Notes
        -----
        This function improves the robustness of the DIIS procedure in
        the event of linear dependencies.
        """

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

        xnew = 0.0
        for i, ci in enumerate(c[1:]):
            xnew += self.get_vec(i) * ci

        return xnew


class SilentSCF:
    """
    Context manager to shut PySCF's SCF classes up.

    Parameters
    ----------
    mf : pyscf.scf.SCF
        SCF object to silence.
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


def list_union(*args):
    """
    Find the union of a list of lists, with the elements sorted
    by their first occurrence.

    Parameters
    ----------
    args : list of list
        Lists to find the union of.

    Returns
    -------
    out : list
        Union of the lists.
    """

    cache = set()
    out = []
    for arg in args:
        for x in arg:
            if x not in cache:
                cache.add(x)
                out.append(x)

    return out
