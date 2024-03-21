"""
******************************
momentGW: Moment-conserving GW
******************************

This repository contains the code and implementation for the paper
"A 'moment-conesrving' reformulation of GW theory".

Installation
------------

Install Vayesta, which includes other dependencies such as PySCF
and NumPy:

    git clone git@github.com:BoothGroup/Vayesta.git
    python -m pip install . --user

Install moment-conserving Dyson equation solver:

    git clone git@github.com:BoothGroup/dyson.git

Clone the momentGW repository:

    git clone git@github.com:BoothGroup/momentGW.git --depth 1
    python -m pip install . --user

"""

import sys
import logging

__version__ = "1.0.0"


# --- Check dependencies

try:
    import pyscf
except ImportError:
    raise ImportError("Missing dependency: https://github.com/pyscf/pyscf")

try:
    from dyson import MBLSE
except ImportError:
    raise ImportError("Missing dependency: https://github.com/BoothGroup/dyson")


# --- Patch `dyson` to suppress logging

import dyson

dyson.default_log.setLevel(logging.CRITICAL)


# --- Imports

from momentGW import logging
from momentGW.logging import init_logging, console, dump_times

from momentGW.tda import dTDA
from momentGW.rpa import dRPA

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
