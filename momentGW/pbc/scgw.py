"""
Spin-restricted self-consistent GW via self-energy moment constraitns
for periodic systems.
"""

import numpy as np
from pyscf import lib
from pyscf.agf2 import GreensFunction, mpi_helper
from pyscf.ao2mo import _ao2mo
from pyscf.lib import logger

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

    def init_gf(self, mo_energy=None):
        """Initialise the mean-field Green's function.

        Parameters
        ----------
        mo_energy : numpy.ndarray, optional
            Molecular orbital energies at each k-point. Default value is
            `self.mo_energy`.

        Returns
        -------
        gf : tuple of GreensFunction
            Mean-field Green's function at each k-point.
        """

        if mo_energy is None:
            mo_energy = self.mo_energy

        gf = []
        for k in self.kpts.loop(1):
            chempot = 0.5 * (mo_energy[k][self.nocc[k] - 1] + mo_energy[k][self.nocc[k]])
            gf.append(GreensFunction(mo_energy[k], np.eye(self.nmo), chempot=chempot))

        return gf

    check_convergence = evKGW.check_convergence
