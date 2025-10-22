"""
momentGW
========

The `momentGW` package implements a wide range of GW methods according
to the moment-conserving formalism.

Examples
--------
Examples of usage can be found in the `examples` directory of the
repository::

    $ python examples/00-moment_order.py

Notes
-----
Publications using `momentGW` should cite [1]_.

References
----------
.. [1] C. J. C. Scott, O. J. Backhouse, and G. H. Booth, 158, 12, 2023.
"""

import importlib
import logging
import sys

__version__ = "1.0.0"


# --- Patch `dyson` to suppress logging

import dyson

dyson.default_log.setLevel(logging.CRITICAL)


# --- Imports

from momentGW.logging import console, dump_times, init_logging


def __getattr__(name):
    """Import handler to allow imports of all solvers from the top-level package without circular
    imports."""
    if name.endswith("GW") or name.endswith("BSE"):
        path = ["momentGW"]
        if "K" in name:
            path.append("pbc")
        if "U" in name:
            path.append("uhf")
        path.append(name.replace("K", "").replace("U", "").replace("cp", "").lower())
        return getattr(importlib.import_module(".".join(path)), name)
    else:
        raise AttributeError(f"module 'momentGW' has no attribute '{name}'")
