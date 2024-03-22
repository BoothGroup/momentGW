"""
k-points helper utilities.
"""

import itertools

import numpy as np
import scipy.linalg
from pyscf import lib
from pyscf.pbc.lib import kpts_helper

from momentGW import mpi_helper, util

# TODO make sure this is rigorous


def allow_single_kpt(output_is_kpts=False):
    """
    Decorate functions to allow `kpts` arguments to be passed as a single
    k-point.

    Parameters
    ----------
    output_is_kpts : bool, optional
        Whether the output of the function is a k-point. Default value
        is `False`.
    """

    def decorator(func):
        def wrapper(self, kpts, *args, **kwargs):
            shape = kpts.shape
            kpts = kpts.reshape(-1, 3)
            res = func(self, kpts, *args, **kwargs)
            if output_is_kpts:
                return res.reshape(shape)
            else:
                return res

        return wrapper

    return decorator


class KPoints:
    """Helper class for k-points.

    Parameters
    ----------
    cell : pyscf.pbc.gto.Cell
        Unit cell.
    kpts : numpy.ndarray
        Array of k-points.
    tol : float, optional
        Threshold for determining if two k-points are equal. Default
        value is `1e-8`.
    wrap_around : bool, optional
        Whether to wrap k-points around the first Brillouin zone. Default
        value is `True`.
    """

    def __init__(self, cell, kpts, tol=1e-8, wrap_around=True):
        self.cell = cell
        self.tol = tol

        if not isinstance(kpts, np.ndarray):
            kpts = kpts.kpts

        if wrap_around:
            kpts = self.wrap_around(kpts)
        self._kpts = kpts

        self._kconserv = kpts_helper.get_kconserv(cell, kpts)
        self._kpts_hash = {self.hash_kpts(kpt): k for k, kpt in enumerate(self._kpts)}

    def member(self, kpt):
        """
        Find the index of the k-point in the k-point list.

        Parameters
        ----------
        kpt : numpy.ndarray
            Array of the k-point.

        Returns
        -------
        index : int
            Index of the k-point.
        """
        if kpt not in self:
            raise ValueError(f"{kpt} is not in list")
        return self._kpts_hash[self.hash_kpts(kpt)]

    def index(self, kpt):
        """
        Alias for `member`.

        Parameters
        ----------
        kpt : numpy.ndarray
            Array of the k-point.

        Returns
        -------
        index : int
            Index of the k-point.
        """
        return self.member(kpt)

    @allow_single_kpt(output_is_kpts=True)
    def get_scaled_kpts(self, kpts):
        """
        Convert absolute k-points to scaled k-points for the current
        cell.

        Parameters
        ----------
        kpts : numpy.ndarray
            Array of absolute k-points.

        Returns
        -------
        scaled_kpts : numpy.ndarray
            Array of scaled k-points.
        """
        return self.cell.get_scaled_kpts(kpts)

    @allow_single_kpt(output_is_kpts=True)
    def get_abs_kpts(self, kpts):
        """
        Convert scaled k-points to absolute k-points for the current
        cell.

        Parameters
        ----------
        kpts : numpy.ndarray
            Array of scaled k-points.

        Returns
        -------
        abs_kpts : numpy.ndarray
            Array of absolute k-points.
        """
        return self.cell.get_abs_kpts(kpts)

    @allow_single_kpt(output_is_kpts=True)
    def wrap_around(self, kpts, window=(-0.5, 0.5)):
        """
        Handle the wrapping of k-points into the first Brillouin zone.

        Parameters
        ----------
        kpts : numpy.ndarray
            Array of absolute k-points.
        window : tuple, optional
            Window within which to contain scaled k-points. Default value
            is `(-0.5, 0.5)`.

        Returns
        -------
        wrapped_kpts : numpy.ndarray
            Array of wrapped k-points.
        """

        kpts = self.get_scaled_kpts(kpts) % 1.0
        kpts = lib.cleanse(kpts, axis=0, tol=self.tol)
        kpts = kpts.round(decimals=self.tol_decimals) % 1.0

        kpts[kpts < window[0]] += 1.0
        kpts[kpts >= window[1]] -= 1.0

        kpts = self.get_abs_kpts(kpts)

        return kpts

    @allow_single_kpt(output_is_kpts=False)
    def hash_kpts(self, kpts):
        """
        Convert k-points to a unique, hashable representation.

        Parameters
        ----------
        kpts : numpy.ndarray
            Array of absolute k-points.

        Returns
        -------
        hash_kpts : tuple
            Hashable representation of k-points.
        """
        return tuple(np.rint(kpts / (self.tol)).ravel().astype(int))

    @property
    def tol_decimals(self):
        """Convert the tolerance into a number of decimal places.

        Returns
        -------
        tol_decimals : int
            Number of decimal places.
        """
        return int(-np.log10(self.tol + 1e-16)) + 2

    def conserve(self, ki, kj, kk):
        """
        Get the index of the k-point that conserves momentum.

        Parameters
        ----------
        ki, kj, kk : int
            Indices of the k-points.

        Returns
        -------
        kconserv : int
            Index of the k-point that conserves momentum.
        """
        return self._kconserv[ki, kj, kk]

    def loop(self, depth, mpi=False):
        """
        Iterate over all combinations of k-points up to a given depth.

        Parameters
        ----------
        depth : int
            Depth of the loop.
        mpi : bool, optional
            Whether to split the loop over MPI processes. Default value
            is `False`.

        Yields
        ------
        kpts : tuple
            Tuple of k-point indices.
        """

        if depth == 1:
            seq = range(len(self))
        else:
            seq = itertools.product(range(len(self)), repeat=depth)

        if mpi:
            size = len(self) * depth
            split = lambda x: x * size // mpi_helper.size

            p0 = split(mpi_helper.rank)
            p1 = size if mpi_helper.rank == (mpi_helper.size - 1) else split(mpi_helper.rank + 1)

            seq = itertools.islice(seq, p0, p1)

        yield from seq

    def loop_size(self, depth=1):
        """
        Return the size of `loop`. Without MPI, this is equivalent to
        `len(self)**depth`.

        Parameters
        ----------
        depth : int, optional
            Depth of the loop. Default value is `1`.

        Returns
        -------
        size : int
            Size of the loop.
        """

        size = len(self) * depth
        split = lambda x: x * size // mpi_helper.size

        p0 = split(mpi_helper.rank)
        p1 = size if mpi_helper.rank == (mpi_helper.size - 1) else split(mpi_helper.rank + 1)

        return p1 - p0

    @allow_single_kpt(output_is_kpts=False)
    def is_zero(self, kpts):
        """
        Check if the k-point is zero.

        Parameters
        ----------
        kpts : numpy.ndarray
            Array of absolute k-points.

        Returns
        -------
        is_zero : bool
            Whether the k-point is zero.
        """
        return np.max(np.abs(kpts)) < self.tol

    @property
    def kmesh(self):
        """Guess the k-mesh.

        Returns
        -------
        kmesh : list
            Size of the k-mesh in each direction.
        """
        kpts = self.get_scaled_kpts(self._kpts).round(self.tol_decimals)
        kmesh = [len(np.unique(kpts[:, i])) for i in range(3)]
        return kmesh

    def translation_vectors(self):
        """
        Build translation vectors to construct supercell of which the
        gamma point is identical to the k-point mesh of the primitive
        cell.

        Returns
        -------
        r_vec_abs : numpy.ndarray
            Array of translation vectors.
        """

        kmesh = self.kmesh

        r_rel = [np.arange(kmesh[i]) for i in range(3)]
        r_vec_rel = lib.cartesian_prod(r_rel)
        r_vec_abs = np.dot(r_vec_rel, self.cell.lattice_vectors())

        return r_vec_abs

    def interpolate(self, other, fk):
        """
        Interpolate a function `f` from the current grid of k-points to
        those of `other`. Input must be in a localised basis, i.e. AOs.

        Parameters
        ----------
        other : KPoints
            The k-points to interpolate to.
        fk : numpy.ndarray
            The function to interpolate, expressed on the current
            k-point grid. Must be a matrix-valued array expressed in
            k-space, *in a localised basis*.

        Returns
        -------
        f : numpy.ndarray
            The interpolated function, expressed on the new k-point grid.
        """

        if len(other) % len(self):
            raise ValueError(
                "Size of destination k-point mesh must be divisible by the size of the source "
                "k-point mesh for interpolation."
            )
        nimg = len(other) // len(self)
        nao = fk.shape[-1]

        r_vec_abs = self.translation_vectors()
        kR = np.exp(1.0j * np.dot(self._kpts, r_vec_abs.T)) / np.sqrt(len(r_vec_abs))

        r_vec_abs = other.translation_vectors()
        kL = np.exp(1.0j * np.dot(other._kpts, r_vec_abs.T)) / np.sqrt(len(r_vec_abs))

        # k -> bvk
        fg = util.einsum("kR,kij,kS->RiSj", kR, fk, kR.conj())
        if np.max(np.abs(fg.imag)) > 1e-6:
            raise ValueError("Interpolated function has non-zero imaginary part.")
        fg = fg.real
        fg = fg.reshape(len(self) * nao, len(self) * nao)

        # tile in bvk
        fg = scipy.linalg.block_diag(*[fg for i in range(nimg)])

        # bvk -> k
        fg = fg.reshape(len(other), nao, len(other), nao)
        fl = util.einsum("kR,RiSj,kS->kij", kL.conj(), fg, kL)

        return fl

    def __array__(self):
        """
        Get the k-points as a numpy array.
        """
        return np.asarray(self._kpts)

    @property
    def T(self):
        """
        Get the transpose of the k-points.
        """
        return self.__array__().T

    def __getitem__(self, index):
        """
        Get the k-point at the given index.

        Parameters
        ----------
        index : int
            Index of the k-point.

        Returns
        -------
        kpt : numpy.ndarray
            Array of the k-point.
        """
        return self._kpts[index]

    def __iter__(self):
        """
        Iterate over the k-points.
        """
        return iter(self._kpts)

    def __contains__(self, kpt):
        """
        Check if the k-point is in the k-point list.

        Parameters
        ----------
        kpt : numpy.ndarray
            Array of the k-point.

        Returns
        -------
        is_in : bool
            Whether the k-point is in the list.
        """
        return self.hash_kpts(kpt) in self._kpts_hash

    def __len__(self):
        """
        Get the number of k-points.
        """
        return len(self._kpts)

    def __eq__(self, other):
        """
        Check if two k-point lists are equal to within `self.tol`.

        Parameters
        ----------
        other : KPoints or numpy.ndarray
            The other k-point list. If a `numpy.ndarray` is given, it
            is converted to a `KPoints` object, complete with the wrap
            around handling.

        Returns
        -------
        is_equal : bool
            Whether the two k-point lists are equal to within
            `self.tol`. Uses the hashes according to `KPoints.hash_kpts`.
        """
        if not isinstance(other, KPoints):
            other = KPoints(self.cell, other, tol=self.tol)
        if len(self) != len(other):
            return False
        return self.hash_kpts(self._kpts) == other.hash_kpts(other._kpts)

    def __ne__(self, other):
        """
        Check if two k-point lists are not equal to within `self.tol`.
        """
        return not self.__eq__(other)

    def __repr__(self):
        """
        Get a string representation of the k-points.
        """
        return repr(self._kpts)

    def __str__(self):
        """
        Get a string representation of the k-points.
        """
        return str(self._kpts)
