"""
Spin-unrestricted self-consistent GW via self-energy moment constraints
for molecular systems.
"""

import numpy as np
from pyscf.lib import logger

from momentGW import util
from momentGW.scgw import scGW
from momentGW.uhf import UGW, evUGW


class scUGW(UGW, scGW):
    __doc__ = scGW.__doc__.replace("Spin-restricted", "Spin-unrestricted", 1)

    _opts = util.list_union(UGW._opts, scGW._opts)

    @property
    def name(self):
        """Method name."""
        polarizability = self.polarizability.upper().replace("DTDA", "dTDA").replace("DRPA", "dRPA")
        return f"{polarizability}-UG{'0' if self.g0 else ''}W{'0' if self.w0 else ''}"

    check_convergence = evUGW.check_convergence
