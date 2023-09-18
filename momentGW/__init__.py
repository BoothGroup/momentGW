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


# --- Imports

from momentGW.tda import TDA
from momentGW.rpa import RPA
from momentGW.gw import GW
from momentGW.bse import BSE, cpBSE
from momentGW.evgw import evGW
from momentGW.scgw import scGW
from momentGW.qsgw import qsGW
from momentGW.pbc.gw import KGW
from momentGW.pbc.evgw import evKGW
from momentGW.pbc.qsgw import qsKGW
from momentGW.pbc.scgw import scKGW
