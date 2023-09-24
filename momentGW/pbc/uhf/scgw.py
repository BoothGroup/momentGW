"""
Spin-unrestricted self-consistent GW via self-energy moment constraints
for periodic systems.
"""

import numpy as np
from pyscf.lib import logger

from momentGW import util
from momentGW.pbc.scgw import scKGW
from momentGW.pbc.uhf.evgw import evKUGW
from momentGW.pbc.uhf.gw import KUGW
from momentGW.uhf.scgw import scUGW


class scKUGW(KUGW, scKGW, scUGW):
    __doc__ = scKGW.__doc__.replace("Spin-restricted", "Spin-unrestricted")

    _opts = util.list_union(scKGW._opts, scKGW._opts, scUGW._opts)

    @property
    def name(self):
        """Method name."""
        polarizability = self.polarizability.upper().replace("DTDA", "dTDA").replace("DRPA", "dRPA")
        return f"{polarizability}-scKUG{'0' if self.g0 else ''}W{'0' if self.w0 else ''}"

    check_convergence = evKUGW.check_convergence
