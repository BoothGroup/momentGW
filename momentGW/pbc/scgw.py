"""
Spin-restricted self-consistent GW via self-energy moment constraitns
for periodic systems.
"""

from momentGW import util
from momentGW.pbc.evgw import evKGW
from momentGW.pbc.gw import KGW
from momentGW.scgw import scGW


class scKGW(KGW, scGW):
    __doc__ = scGW.__doc__.replace("molecules", "periodic systems", 1)

    _opts = util.list_union(KGW._opts, scGW._opts)

    @property
    def name(self):
        return "scKG%sW%s" % ("0" if self.g0 else "", "0" if self.w0 else "")

    check_convergence = evKGW.check_convergence
