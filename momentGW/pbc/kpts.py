"""
k-points helper utilities.
"""

import itertools

import numpy as np
from pyscf import lib
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

    def loop(self, depth):
        """
        Iterate over all combinations of k-points up to a given depth.
        """
        return itertools.product(enumerate(self), repeat=depth)

    @allow_single_kpt(output_is_kpts=False)
    def is_zero(self, kpts):
        """
        Check if the k-point is zero.
        """
        return np.max(np.abs(kpts)) < self.tol

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
