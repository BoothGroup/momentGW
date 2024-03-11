"""
Spin-unrestricted Fock matrix self-consistent GW via self-energy moment
constraints for periodic systems.
"""

from momentGW import util
from momentGW.pbc.fsgw import fsKGW
from momentGW.pbc.uhf.gw import KUGW
from momentGW.pbc.uhf.qsgw import qsKUGW
from momentGW.uhf.fsgw import fsUGW


class fsKUGW(KUGW, fsKGW, fsUGW):  # noqa: D101
    __doc__ = fsKGW.__doc__.replace("Spin-restricted", "Spin-unrestricted", 1)

    # --- Default fsKUGW options

    solver = KUGW

    _opts = util.list_union(KUGW._opts, fsKGW._opts, fsUGW._opts)

    @property
    def name(self):
        """Method name."""
        polarizability = self.polarizability.upper().replace("DTDA", "dTDA").replace("DRPA", "dRPA")
        return f"{polarizability}-fsKUGW"

    project_basis = staticmethod(qsKUGW.project_basis)
    self_energy_to_moments = staticmethod(qsKUGW.self_energy_to_moments)
    check_convergence = qsKUGW.check_convergence
