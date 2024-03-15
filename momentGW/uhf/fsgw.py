"""
Spin-unrestricted Fock matrix self-consistent GW via self-energy moment
constraints for molecular systems.
"""

from momentGW import util
from momentGW.fsgw import fsGW
from momentGW.uhf.gw import UGW
from momentGW.uhf.qsgw import qsUGW


class fsUGW(UGW, fsGW):  # noqa: D101
    __doc__ = fsGW.__doc__.replace("Spin-restricted", "Spin-unrestricted", 1)

    # --- Default fsUGW options

    solver = UGW

    _opts = util.list_union(UGW._opts, fsGW._opts)

    @property
    def name(self):
        """Method name."""
        polarizability = self.polarizability.upper().replace("DTDA", "dTDA").replace("DRPA", "dRPA")
        return f"{polarizability}-fsUGW"

    project_basis = staticmethod(qsUGW.project_basis)
    self_energy_to_moments = staticmethod(qsUGW.self_energy_to_moments)
    check_convergence = qsUGW.check_convergence
