"""
Spin-restricted one-shot GW via self-energy moment constraints for
periodic systems.
"""

import numpy as np
from dyson import MBLSE, MixedMBLSE, NullLogger
from pyscf import lib
from pyscf.agf2 import GreensFunction, SelfEnergy
from pyscf.lib import logger
from pyscf.pbc import scf

from momentGW.gw import GW
from momentGW.pbc.base import BaseKGW
from momentGW.pbc.fock import fock_loop, minimize_chempot, search_chempot
from momentGW.pbc.ints import KIntegrals
from momentGW.pbc.tda import TDA


class KGW(BaseKGW, GW):
    __doc__ = BaseKGW.__doc__.format(
        description="Spin-restricted one-shot GW via self-energy moment constraints for "
        + "periodic systems.",
        extra_parameters="",
    )

    @property
    def name(self):
        return "KG0W0"

    def ao2mo(self):
        """Get the integrals."""

        integrals = KIntegrals(
            self.with_df,
            self.kpts,
            self.mo_coeff,
            self.mo_occ,
            compression=self.compression,
            compression_tol=self.compression_tol,
            store_full=self.has_fock_loop,
        )
        integrals.transform()

        return integrals

    def build_se_moments(self, nmom_max, integrals, **kwargs):
        """Build the moments of the self-energy.

        Parameters
        ----------
        nmom_max : int
            Maximum moment number to calculate.
        integrals : KIntegrals
            Density-fitted integrals.

        See functions in `momentGW.rpa` for `kwargs` options.

        Returns
        -------
        se_moments_hole : numpy.ndarray
            Moments of the hole self-energy at each k-point. If
            `self.diagonal_se`, non-diagonal elements are set to zero.
        se_moments_part : numpy.ndarray
            Moments of the particle self-energy at each k-point. If
            `self.diagonal_se`, non-diagonal elements are set to zero.
        """

        if self.polarizability == "dtda":
            tda = TDA(self, nmom_max, integrals, **kwargs)
            return tda.kernel()
        else:
            raise NotImplementedError

    def solve_dyson(self, se_moments_hole, se_moments_part, se_static, integrals=None):
        """Solve the Dyson equation due to a self-energy resulting
        from a list of hole and particle moments, along with a static
        contribution.

        Also finds a chemical potential best satisfying the physical
        number of electrons. If `self.optimise_chempot`, this will
        shift the self-energy poles relative to the Green's function,
        which is a partial self-consistency that better conserves the
        particle number.

        If `self.fock_loop`, this function will also require that the
        outputted Green's function is self-consistent with respect to
        the corresponding density and Fock matrix.

        Parameters
        ----------
        se_moments_hole : numpy.ndarray
            Moments of the hole self-energy at each k-point.
        se_moments_part : numpy.ndarray
            Moments of the particle self-energy at each k-point.
        se_static : numpy.ndarray
            Static part of the self-energy at each k-point.
        integrals : KIntegrals, optional
            Density-fitted integrals.  Required if `self.fock_loop`
            is `True`.  Default value is `None`.

        Returns
        -------
        gf : list of pyscf.agf2.GreensFunction
            Green's function at each k-point.
        se : list of pyscf.agf2.SelfEnergy
            Self-energy at each k-point.
        """

        nlog = NullLogger()

        se = []
        gf = []
        for k in self.kpts.loop(1):
            solver_occ = MBLSE(se_static[k], np.array(se_moments_hole[k]), log=nlog)
            solver_occ.kernel()

            solver_vir = MBLSE(se_static[k], np.array(se_moments_part[k]), log=nlog)
            solver_vir.kernel()

            solver = MixedMBLSE(solver_occ, solver_vir)
            e_aux, v_aux = solver.get_auxiliaries()
            se.append(SelfEnergy(e_aux, v_aux))

            logger.debug(
                self,
                "Error in moments [kpt %d]: occ = %.6g  vir = %.6g",
                k,
                *self.moment_error(se_moments_hole[k], se_moments_part[k], se[k]),
            )

            gf.append(se[k].get_greens_function(se_static[k]))

        if self.optimise_chempot:
            se, opt = minimize_chempot(se, se_static, sum(self.nocc) * 2)

        if self.fock_loop:
            gf, se, conv = fock_loop(self, gf, se, integrals=integrals, **self.fock_opts)

        w = [g.energy for g in gf]
        v = [g.coupling for g in gf]
        cpt, error = search_chempot(w, v, self.nmo, sum(self.nocc) * 2)

        for k in self.kpts.loop(1):
            se[k].chempot = cpt
            gf[k].chempot = cpt
            logger.info(self, "Error in number of electrons [kpt %d]: %.5g", k, error)

        return gf, se

    def make_rdm1(self, gf=None):
        """Get the first-order reduced density matrix at each k-point."""

        if gf is None:
            gf = self.gf
        if gf is None:
            gf = [GreensFunction(self.mo_energy, np.eye(self.nmo))]

        return np.array([g.make_rdm1() for g in gf])
