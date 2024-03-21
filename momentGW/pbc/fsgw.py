"""
Spin-restricted Fock matrix self-consistent GW via self-energy moment
constraints for periodic systems.
"""

from momentGW import util
from momentGW.fsgw import fsGW
from momentGW.pbc.gw import KGW
from momentGW.pbc.qsgw import qsKGW


class fsKGW(KGW, fsGW):  # noqa: D101
    __doc__ = fsGW.__doc__.replace("molecules", "periodic systems", 1)

    # --- Default fsKGW options

    solver = KGW

    _opts = util.list_union(KGW._opts, fsGW._opts)

    project_basis = staticmethod(qsKGW.project_basis)
    self_energy_to_moments = staticmethod(qsKGW.self_energy_to_moments)
    check_convergence = qsKGW.check_convergence

    @property
    def name(self):
        """Method name."""
        polarizability = self.polarizability.upper().replace("DTDA", "dTDA").replace("DRPA", "dRPA")
        return f"{polarizability}-fsKGW"
