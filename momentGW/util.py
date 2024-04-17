"""Utility functions.
"""

import time

import numpy as np
import scipy.linalg
from pyscf import __config__ as _pyscf_config
from pyscf import lib

# Define the size of problem to fall back on NumPy
NUMPY_EINSUM_SIZE = 2000


class Timer:
    """Timer class."""

    def __init__(self):
        self.t_init = time.perf_counter()
        self.t_prev = time.perf_counter()
        self.t_curr = time.perf_counter()

    def lap(self):
        """Return the time since the last call to `lap`.

        Returns
        -------
        lap : float
            Lap time.
        """
        self.t_prev, self.t_curr = self.t_curr, time.perf_counter()
        return self.t_curr - self.t_prev

    def total(self):
        """Return the total time since initialization.

        Returns
        -------
        total : float
            Total time.
        """
        return time.perf_counter() - self.t_init

    @staticmethod
    def format_time(seconds, precision=2):
        """Return a formatted time.

        Parameters
        ----------
        seconds : float
            Time in seconds.
        precision : int, optional
            Number of time units to display. Default is `2`.

        Returns
        -------
        formatted : str
            Formatted time.
        """

        # Get the time in hours, minutes, seconds, and milliseconds
        seconds, milliseconds = divmod(seconds, 1)
        milliseconds *= 1000
        minutes, seconds = divmod(seconds, 60)
        hours, minutes = divmod(minutes, 60)

        # Format the time
        out = []
        if hours:
            out.append("%3d h" % hours)
        if minutes:
            out.append("%2d m" % minutes)
        if seconds:
            out.append("%2d s" % seconds)
        if milliseconds:
            out.append("%3d ms" % milliseconds)

        return " ".join(out[-max(precision, len(out)) :])

    __call__ = lap


class DIIS(lib.diis.DIIS):
    """
    Direct inversion of the iterative subspace (DIIS).

    Notes
    -----
    For some reason, the default pyscf DIIS object can result in fully
    linearly dependent error vectors in high-moment self-consistent
    calculations. This class is a drop-in replacement with a fallback
    in this case.

    See Also
    --------
    pyscf.lib.diis.DIIS : PySCF DIIS object which this class extends.
    """

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

        # Check if the object is complex
        if not np.iscomplexobj(x):
            return self.update(x, xerr=xerr)

        # Get the shape
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

        See Also
        --------
        pyscf.lib.diis.DIIS.extrapolate :
            PySCF DIIS extrapolation which this function refactors.
        """

        # Get the number of vectors
        if nd is None:
            nd = self.get_num_vec()
        if nd == 0:
            raise RuntimeError("No vector found in DIIS object.")

        # Get the Hessian
        h = self._H[: nd + 1, : nd + 1]
        g = np.zeros(nd + 1, h.dtype)
        g[0] = 1

        # Solve the linear system
        w, v = scipy.linalg.eigh(h)
        if np.any(abs(w) < 1e-14):
            idx = abs(w) > 1e-14
            c = np.dot(v[:, idx] * (1.0 / w[idx]), np.dot(v[:, idx].T.conj(), g))
        else:
            try:
                c = np.linalg.solve(h, g)
            except np.linalg.linalg.LinAlgError as e:
                raise np.linalg.linalg.LinAlgError("DIIS matrix is singular.") from e

        # Check for linear dependencies
        if np.all(abs(c) < 1e-14):
            raise np.linalg.linalg.LinAlgError("DIIS vectors are fully linearly dependent.")

        # Extrapolate
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
        self._cache = {}

    def __enter__(self):
        """
        Return the SCF object with verbosity set to zero.
        """

        self._cache["config"] = _pyscf_config.VERBOSE
        _pyscf_config.VERBOSE = 0

        self._cache["mol"] = self.mf.mol.verbose
        self.mf.mol.verbose = 0

        self._cache["mf"] = self.mf.verbose
        self.mf.verbose = 0

        if getattr(self.mf, "with_df", None):
            self._cache["df"] = self.mf.with_df.verbose
            self.mf.with_df.verbose = 0

        return self.mf

    def __exit__(self, exc_type, exc_value, traceback):
        """
        Reset the verbosity of the SCF object.
        """
        _pyscf_config.VERBOSE = self._cache["config"]
        self.mf.mol.verbose = self._cache["mol"]
        self.mf.verbose = self._cache["mf"]
        if getattr(self.mf, "with_df", None):
            self.mf.with_df.verbose = self._cache["df"]


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


def dict_union(*args):
    """
    Find the union of a list of dictionaries, preserving the order
    of the first occurrence of each key.

    Parameters
    ----------
    args : list of dict
        Dictionaries to find the union of.

    Returns
    -------
    out : dict
        Union of the dictionaries.
    """
    cache = set()
    out = type(args[0])() if len(args) else {}
    for arg in args:
        for x in arg:
            if x not in cache:
                cache.add(x)
                out[x] = arg[x]
    return out


def build_1h1p_energies(mo_energy, mo_occ):
    r"""
    Construct an array of 1h1p energies where elements are

    .. math::
       \\Delta_{ij} = \\epsilon_i - \\epsilon_j

    Parameters
    ----------
    mo_energy : numpy.ndarray or tuple of numpy.ndarray
        Molecular orbital energies. If a tuple, the first element
        is used for occupied orbitals and the second element is used
        for virtual orbitals.
    mo_occ : numpy.ndarray or tuple of numpy.ndarray
        Molecular orbital occupancies. If a tuple, the first element
        is used for occupied orbitals and the second element is used
        for virtual orbitals.

    Returns
    -------
    d : numpy.ndarray
        1h1p energies.
    """

    # Check if the input is a tuple
    if not isinstance(mo_energy, tuple):
        mo_energy = (mo_energy, mo_energy)
    if not isinstance(mo_occ, tuple):
        mo_occ = (mo_occ, mo_occ)

    # Get the occupied and virtual energies
    e_occ = mo_energy[0][mo_occ[0] > 0]
    e_vir = mo_energy[1][mo_occ[1] == 0]

    # Construct the energy differences
    d = lib.direct_sum("a-i->ia", e_vir, e_occ)

    return d


def _contract(subscript, *args, **kwargs):
    """Contract a pair of terms in an `einsum`-like fashion.

    Parameters
    ----------
    subscript : str
        Subscript notation for the contraction.
    args : list of numpy.ndarray
        Arrays to contract.
    kwargs : dict
        Additional arguments to pass to `numpy.einsum`.

    Returns
    -------
    out : numpy.ndarray
        Contracted array.
    """

    # Unpack the arguments
    a, b = args
    a = np.asarray(a, order="A")
    b = np.asarray(b, order="A")

    # Fall back function
    def _fallback():
        numpy_kwargs = kwargs.copy()
        if "optimize" not in kwargs:
            numpy_kwargs["optimize"] = True
        return np.einsum(subscript, a, b, **numpy_kwargs)

    # Check if the problem is small enough to use NumPy
    if min(a.size, b.size) < NUMPY_EINSUM_SIZE:
        return _fallback()

    # Make sure the problem can be dispatched as a DGEMM
    indices = subscript.replace(",", "").replace("->", "")
    if any(indices.count(x) != 2 for x in set(indices)):
        return _fallback()

    # Get the characters for each input and output
    inp, out, args = np.core.einsumfunc._parse_einsum_input((subscript, a, b))
    inp_a, inp_b = inps = inp.split(",")
    assert len(inps) == len(args) == 2
    assert all(len(inp) == arg.ndim for inp, arg in zip(inps, args))

    # Check for internal traces
    if any(len(inp) != len(set(inp)) for inp in inps):
        # FIXME this can be consumed and then re-call _contract
        return _fallback()

    # Find the dummy indices
    dummy = set(inp_a) & set(inp_b)
    if not dummy or inp_a == dummy or inp_b == dummy:
        return _fallback()

    # Find the index sizes
    ranges = {}
    for inp, arg in zip(inps, args):
        for i, s in zip(inp, arg.shape):
            if i in ranges:
                if ranges[i] != s:
                    raise ValueError(
                        f"Index size mismatch: {subscript} with A={a.shape} and B={b.shape}."
                    )
            ranges[i] = s

    # Align indices for DGEMM
    inp_at = list(inp_a)
    inp_bt = list(inp_b)
    inner_shape = 1
    for i, n in enumerate(sorted(dummy)):
        j = len(inp_at) - 1
        inp_at.insert(j, inp_at.pop(inp_at.index(n)))
        inp_bt.insert(i, inp_bt.pop(inp_bt.index(n)))
        inner_shape *= ranges[n]

    # Find transposes
    order_a = [inp_a.index(idx) for idx in inp_at]
    order_b = [inp_b.index(idx) for idx in inp_bt]

    # Get the shape and transpose for the output
    shape_ct = []
    inp_ct = []
    for idx in inp_at:
        if idx in dummy:
            break
        shape_ct.append(ranges[idx])
        inp_ct.append(idx)
    for idx in inp_bt:
        if idx in dummy:
            continue
        shape_ct.append(ranges[idx])
        inp_ct.append(idx)
    order_ct = [inp_ct.index(idx) for idx in out]

    # If any dimension has size zero, return here
    if a.size == 0 or b.size == 0:
        shape_c = [shape_ct[i] for i in order_ct]
        if kwargs.get("out", None):
            return kwargs["out"].reshape(shape_c)
        else:
            return np.zeros(shape_c, dtype=np.result_type(a, b))

    # Apply transposes
    at = a.transpose(order_a)
    bt = b.transpose(order_b)

    # Find optimal memory layout
    at = np.asarray(at.reshape(-1, inner_shape), order="F" if at.flags.f_contiguous else "C")
    bt = np.asarray(bt.reshape(inner_shape, -1), order="F" if bt.flags.f_contiguous else "C")

    # Get the output buffer
    shape_ct_flat = (at.shape[0], bt.shape[1])
    if kwargs.get("out", None):
        order_c = [out.index(idx) for idx in inp_ct]
        out = kwargs["out"].transpose(order_c)
        out = np.asarray(out.reshape(shape_ct_flat), order="C")
    else:
        out = np.empty(shape_ct_flat, dtype=np.result_type(a, b), order="C")

    # Perform the contraction
    ct = lib.dot(at, bt, c=out)

    # Reshape and transpose the output
    ct = ct.reshape(shape_ct, order="A")
    if ct.flags.f_contiguous:
        c = np.asfortranarray(ct.transpose(order_ct))
    elif ct.flags.c_contiguous:
        c = np.ascontiguousarray(ct.transpose(order_ct))
    else:
        c = ct.transpose(order_ct)

    return c


def einsum(*operands, **kwargs):
    """
    Evaluate an Einstein summation convention on the operands.

    Using the Einstein summation convention, many common
    multi-dimensional, linear algebraic array operations can be
    represented in a simple fashion. In *implicit* mode `einsum`
    computes these values.

    In *explicit* mode, `einsum` provides further flexibility to compute
    other array operations that might not be considered classical
    Einstein summation operations, by disabling, or forcing summation
    over specified subscript labels.

    See the `numpy.einsum` documentation for clarification.

    Parameters
    ----------
    operands : list
        Any valid input to `numpy.einsum`.
    out : numpy.ndarray, optional
        If provided, the calculation is done into this array.
    contract : callable, optional
        The function to use for contraction. Default value is
        `_contract`.
    optimize : bool, optional
        If `True`, use the `numpy.einsum_path` to optimize the
        contraction. Default value is `True`.

    Returns
    -------
    output : numpy.ndarray
        The calculation based on the Einstein summation convention.

    See Also
    --------
    numpy.einsum : NumPy's `einsum` function.
    """

    # Parse the input
    inp, out, args = np.core.einsumfunc._parse_einsum_input(operands)
    subscript = f"{inp}->{out}"
    contract = kwargs.pop("contract", _contract)

    # If it's just a transpose, fallback to NumPy
    if len(args) < 2:
        return np.einsum(subscript, *args, **kwargs)

    # If it's a single contraction, call the contract function directly
    if len(args) == 2:
        return contract(subscript, *args, **kwargs)

    # Otherwise, use the `einsum_path` to optimize the contraction
    contractions = np.einsum_path(
        subscript,
        *args,
        optimize=kwargs.get("optimize", True),
        einsum_call=True,
    )[1]

    # Execute the contractions in order
    args = list(args)
    for i, (inds, idx_rm, einsum_str, remaining, _) in enumerate(contractions):
        operands = [args.pop(x) for x in inds]

        # Output should only be provided for the last contraction
        tmp_kwargs = kwargs.copy()
        if i != len(contractions) - 1:
            tmp_kwargs["out"] = None

        # Execute the contraction
        out = contract(einsum_str, *operands, **tmp_kwargs)
        args.append(out)

    return out
