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

from momentGW import energy, util
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

    _opts = util.list_union(BaseKGW._opts, GW._opts)

    @property
    def name(self):
        return "KG0W0"

    def ao2mo(self, transform=True):
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
        if transform:
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

        # Calculate energies
        e_1b = self.energy_hf(gf=gf, integrals=integrals) + self.energy_nuc()
        e_2b_g0 = self.energy_gm(se=se, g0=True)
        logger.info(self, "Energies:")
        logger.info(self, "  One-body:              %15.10g", e_1b)
        logger.info(self, "  Galitskii-Migdal (G0): %15.10g", e_1b + e_2b_g0)
        if not self.polarizability.lower().startswith("thc"):
            # This is N^4
            e_2b = self.energy_gm(gf=gf, se=se, g0=False)
            logger.info(self, "  Galitskii-Migdal (G):  %15.10g", e_1b + e_2b)

        return gf, se

    def make_rdm1(self, gf=None):
        """Get the first-order reduced density matrix at each k-point."""

        if gf is None:
            gf = self.gf
        if gf is None:
            gf = [GreensFunction(e, np.eye(self.nmo)) for e in self.mo_energy]

        return np.array([g.make_rdm1() for g in gf])

    def energy_hf(self, gf=None, integrals=None):
        """Calculate the one-body (Hartree--Fock) energy."""

        if gf is None:
            gf = self.gf
        if integrals is None:
            integrals = self.ao2mo()

        h1e = lib.einsum(
            "kpq,kpi,kqj->kij", self._scf.get_hcore(), self.mo_coeff.conj(), self.mo_coeff
        )
        rdm1 = self.make_rdm1()
        fock = integrals.get_fock(rdm1, h1e)

        e_1b = sum(energy.hartree_fock(rdm1[k], fock[k], h1e[k]) for k in self.kpts.loop(1))
        e_1b /= self.nkpts

        return e_1b.real

    def energy_gm(self, gf=None, se=None, g0=True):
        """Calculate the two-body (Galitskii--Migdal) energy."""

        if gf is None:
            gf = self.gf
        if se is None:
            se = self.se

        if g0:
            e_2b = sum(
                energy.galitskii_migdal_g0(self.mo_energy[k], self.mo_occ[k], se[k])
                for k in self.kpts.loop(1)
            )
        else:
            e_2b = sum(energy.galitskii_migdal(gf[k], se[k]) for k in self.kpts.loop(1))

        # Extra factor for non-self-consistent G
        e_2b *= 0.5

        return e_2b.real

    def interpolate(self, mf, nmom_max):
        """
        Interpolate the object to a new k-point grid, represented by a
        new mean-field object.

        Parameters
        ----------
        mf : pyscf.pbc.scf.KSCF
            Mean-field object on new k-point mesh.
        nmom_max : int
            Maximum moment number to calculate.

        Returns
        -------
        other : __class__
            Interpolated object.
        """

        if len(mf.kpts) % len(self.kpts) != 0:
            raise ValueError("Size of interpolation grid must be a multiple of the old grid.")

        other = self.__class__(mf)
        other.__dict__.update({key: getattr(self, key) for key in self._opts})
        sc = lib.einsum("kpq,kqi->kpi", mf.get_ovlp(), mf.mo_coeff)

        def interp(m):
            # Interpolate part of the moments via the AO basis
            m = lib.einsum("knij,kpi,kqj->knpq", m, self.mo_coeff, self.mo_coeff.conj())
            m = np.stack(
                [self.kpts.interpolate(other.kpts, m[:, n]) for n in range(nmom_max + 1)],
                axis=1,
            )
            m = lib.einsum("knpq,kpi,kqj->knij", m, sc.conj(), sc)
            return m

        # Get the moments of the self-energy on the small k-point grid
        th = np.array([se.get_occupied().moment(range(nmom_max + 1)) for se in self.se])
        tp = np.array([se.get_virtual().moment(range(nmom_max + 1)) for se in self.se])

        # Interpolate the moments
        th = interp(th)
        tp = interp(tp)

        # Get the static self-energy on the fine k-point grid
        integrals = other.ao2mo(transform=False)
        se_static = other.build_se_static(integrals)

        # Solve the Dyson equation on the fine k-point grid
        gf, se = other.solve_dyson(th, tp, se_static, integrals=integrals)

        # Set attributes
        # TODO handle _qp_energy if not None
        other.gf = gf
        other.se = se

        return other

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
