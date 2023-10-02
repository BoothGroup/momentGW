"""
Construct RPA moments with periodic boundary conditions.
"""

import numpy as np
import scipy.special
import scipy.optimize
from pyscf import lib

from momentGW import mpi_helper, util
from momentGW.rpa import dRPA as MoldRPA

class dRPA(MoldRPA):
    """
    Compute the self-energy moments using dTDA and numerical integration
    with periodic boundary conditions.

    Parameters
    ----------
    gw : BaseKGW
        GW object.
    nmom_max : int
        Maximum moment number to calculate.
    mo_energy : dict, optional
        Molecular orbital energies. Keys are "g" and "w" for the Green's
        function and screened Coulomb interaction, respectively.
        If `None`, use `gw.mo_energy` for both. Default value is `None`.
    mo_occ : dict, optional
        Molecular orbital occupancies. Keys are "g" and "w" for the
        Green's function and screened Coulomb interaction, respectively.
        If `None`, use `gw.mo_occ` for both. Default value is `None`.
    """

    def integrate(self, q):
        """Optimise the quadrature and perform the integration for a
        given set of k points.

        Returns
        -------
        integral : numpy.ndarray
            Integral array, including the offset part.
        """

        kpts = self.kpts

        cput0 = (lib.logger.process_clock(), lib.logger.perf_counter())
        lib.logger.info(self.gw, "Performing integration")
        lib.logger.debug(self.gw, "Memory usage: %.2f GB",
                         self._memory_usage())
        p0, p1 = self.mpi_slice(self.nov)

        integrals = np.zeros((self.nkpts, self.nkpts),
                 dtype=object)

        diag_eri = np.zeros((self.nkpts, self.nkpts,self.nov))



    @property
    def naux(self):
        """Number of auxiliaries."""
        return self.integrals.naux

    @property
    def nov(self):
        """Number of ov states in W."""
        return np.multiply.outer(
            [np.sum(occ > 0) for occ in self.mo_occ_w],
            [np.sum(occ == 0) for occ in self.mo_occ_w],
        )

    @property
    def kpts(self):
        """k-points."""
        return self.gw.kpts

    @property
    def nkpts(self):
        """Number of k-points."""
        return self.gw.nkpts
