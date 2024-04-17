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

import sys
import logging
import importlib

__version__ = "1.0.0"


# --- Patch `dyson` to suppress logging

import dyson

dyson.default_log.setLevel(logging.CRITICAL)


# --- Imports

from momentGW.logging import init_logging, console, dump_times

from momentGW.gw import GW
from momentGW.bse import BSE, cpBSE
from momentGW.evgw import evGW
from momentGW.qsgw import qsGW
from momentGW.fsgw import fsGW
from momentGW.scgw import scGW

from momentGW.pbc.gw import KGW
from momentGW.pbc.evgw import evKGW
from momentGW.pbc.qsgw import qsKGW
from momentGW.pbc.fsgw import fsKGW
from momentGW.pbc.scgw import scKGW

from momentGW.uhf.gw import UGW
from momentGW.uhf.evgw import evUGW
from momentGW.uhf.qsgw import qsUGW
from momentGW.uhf.fsgw import fsUGW
from momentGW.uhf.scgw import scUGW

from momentGW.pbc.uhf.gw import KUGW
from momentGW.pbc.uhf.evgw import evKUGW
from momentGW.pbc.uhf.qsgw import qsKUGW
from momentGW.pbc.uhf.fsgw import fsKUGW
from momentGW.pbc.uhf.scgw import scKUGW
