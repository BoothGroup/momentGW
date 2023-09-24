"""
Temporary workaround for `mpi_helper` functions.
"""

from pyscf.agf2.mpi_helper import *

_prange = prange


def prange(start, stop, step):
    try:
        first = next(_prange(start, stop, step))
        empty = False
    except StopIteration:
        empty = True

    if empty:
        yield 0, 0
    else:
        yield from _prange(start, stop, step)
