"""
k-points helper utilities.
"""

import itertools

import numpy as np
import scipy.linalg
from dyson import Lehmann
from pyscf import lib
from pyscf.agf2 import GreensFunction, SelfEnergy, mpi_helper
from pyscf.pbc.lib import kpts_helper

# TODO make sure this is rigorous


def allow_single_kpt(output_is_kpts=False):
    """
    Decorator to allow `kpts` arguments to be passed as a single
    k-point.
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
    def __init__(self, cell, kpts, tol=1e-8, wrap_around=True):
        self.cell = cell
        self.tol = tol

        if wrap_around:
            kpts = self.wrap_around(kpts)
        self._kpts = kpts

        self._kconserv = kpts_helper.get_kconserv(cell, kpts)
        self._kpts_hash = {self.hash_kpts(kpt): k for k, kpt in enumerate(self._kpts)}

    @allow_single_kpt(output_is_kpts=True)
    def get_scaled_kpts(self, kpts):
        """
        Convert absolute k-points to scaled k-points for the current
        cell.
        """
        return self.cell.get_scaled_kpts(kpts)

    @allow_single_kpt(output_is_kpts=True)
    def get_abs_kpts(self, kpts):
        """
        Convert scaled k-points to absolute k-points for the current
        cell.
        """
        return self.cell.get_abs_kpts(kpts)

    @allow_single_kpt(output_is_kpts=True)
    def wrap_around(self, kpts, window=(-0.5, 0.5)):
        """
        Handle the wrapping of k-points into the first Brillouin zone.
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
        """
        return tuple(np.rint(kpts / (self.tol)).ravel().astype(int))

    @property
    def tol_decimals(self):
        """Convert the tolerance into a number of decimal places."""
        return int(-np.log10(self.tol + 1e-16)) + 2

    def conserve(self, ki, kj, kk):
        """
        Get the index of the k-point that conserves momentum.
        """
        return self._kconserv[ki, kj, kk]

    def loop(self, depth, mpi=False):
        """
        Iterate over all combinations of k-points up to a given depth.
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
        """
        return np.max(np.abs(kpts)) < self.tol

    @property
    def kmesh(self):
        """Guess the k-mesh."""
        kpts = self.get_scaled_kpts(self._kpts).round(self.tol_decimals)
        kmesh = [len(np.unique(kpts[:, i])) for i in range(3)]
        return kmesh

    def translation_vectors(self):
        """
        Translation vectors to construct supercell of which the gamma
        point is identical to the k-point mesh of the primitive cell.
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
        fg = lib.einsum("kR,kij,kS->RiSj", kR, fk, kR.conj())
        if np.max(np.abs(fg.imag)) > 1e-6:
            raise ValueError("Interpolated function has non-zero imaginary part.")
        fg = fg.real
        fg = fg.reshape(len(self) * nao, len(self) * nao)

        # tile in bvk
        fg = scipy.linalg.block_diag(*[fg for i in range(nimg)])

        # bvk -> k
        fg = fg.reshape(len(other), nao, len(other), nao)
        fl = lib.einsum("kR,RiSj,kS->kij", kL.conj(), fg, kL)

        return fl

    def member(self, kpt):
        """
        Find the index of the k-point in the k-point list.
        """
        if kpt not in self:
            raise ValueError(f"{kpt} is not in list")
        return self._kpts_hash[self.hash_kpts(kpt)]

    index = member

    def __contains__(self, kpt):
        """
        Check if the k-point is in the k-point list.
        """
        return self.hash_kpts(kpt) in self._kpts_hash

    def __getitem__(self, index):
        """
        Get the k-point at the given index.
        """
        return self._kpts[index]

    def __len__(self):
        """
        Get the number of k-points.
        """
        return len(self._kpts)

    def __iter__(self):
        """
        Iterate over the k-points.
        """
        return iter(self._kpts)

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

    def __array__(self):
        """
        Get the k-points as a numpy array.
        """
        return np.asarray(self._kpts)
