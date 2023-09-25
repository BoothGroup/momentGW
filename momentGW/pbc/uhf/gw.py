"""
Spin-unrestricted one-shot GW via self-energy moment constraints for
periodic systems.
"""

import numpy as np
from dyson import MBLSE, MixedMBLSE, NullLogger
from pyscf import lib
from pyscf.agf2 import GreensFunction, SelfEnergy
from pyscf.lib import logger

from momentGW import energy, util
from momentGW.pbc.fock import minimize_chempot, search_chempot, search_chempot_unconstrained
from momentGW.pbc.gw import KGW
from momentGW.pbc.uhf.base import BaseKUGW
from momentGW.pbc.uhf.fock import fock_loop
from momentGW.pbc.uhf.ints import KUIntegrals
from momentGW.pbc.uhf.tda import dTDA
from momentGW.uhf.gw import UGW


class KUGW(BaseKUGW, KGW, UGW):
    __doc__ = BaseKUGW.__doc__.format(
        description="Spin-unrestricted one-shot GW via self-energy moment constraints for "
        + "periodic systems.",
        extra_parameters="",
    )

    _opts = util.list_union(BaseKUGW._opts, KGW._opts, UGW._opts)

    @property
    def name(self):
        """Method name."""
        polarizability = self.polarizability.upper().replace("DTDA", "dTDA").replace("DRPA", "dRPA")
        return f"{polarizability}-KUG0W0"

    def ao2mo(self, transform=True):
        """Get the integrals object.

        Parameters
        ----------
        transform : bool, optional
            Whether to transform the integrals object.

        Returns
        -------
        integrals : KUIntegrals
            Integrals object.
        """

        # TODO better inheritance
        integrals = KUIntegrals(
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
        integrals : KUIntegrals
            Density-fitted integrals.

        See functions in `momentGW.rpa` for `kwargs` options.

        Returns
        -------
        se_moments_hole : numpy.ndarray
            Moments of the hole self-energy at each k-point for each
            spin channel. If `self.diagonal_se`, non-diagonal elements
            are set to zero.
        se_moments_part : numpy.ndarray
            Moments of the particle self-energy at each k-point for each
            spin channel. If `self.diagonal_se`, non-diagonal elements
            are set to zero.
        """

        # TODO better inheritance
        if self.polarizability.lower() == "dtda":
            tda = dTDA(self, nmom_max, integrals, **kwargs)
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
            Moments of the hole self-energy at each k-point for each
            spin channel.
        se_moments_part : numpy.ndarray
            Moments of the particle self-energy at each k-point for
            each spin channel.
        se_static : numpy.ndarray
            Static part of the self-energy at each k-point for each
            spin channel.
        integrals : KUIntegrals, optional
            Density-fitted integrals.  Required if `self.fock_loop`
            is `True`.  Default value is `None`.

        Returns
        -------
        gf : list of list of pyscf.agf2.GreensFunction
            Green's function at each k-point for each spin channel.
        se : list of list of pyscf.agf2.SelfEnergy
            Self-energy at each k-point for each spin channel.
        """

        nlog = NullLogger()

        se = [[], []]
        gf = [[], []]
        for k in self.kpts.loop(1):
            solver_occ = MBLSE(se_static[0][k], np.array(se_moments_hole[0][k]), log=nlog)
            solver_occ.kernel()

            solver_vir = MBLSE(se_static[0][k], np.array(se_moments_part[0][k]), log=nlog)
            solver_vir.kernel()

            solver = MixedMBLSE(solver_occ, solver_vir)
            e_aux, v_aux = solver.get_auxiliaries()
            se[0].append(SelfEnergy(e_aux, v_aux))

            solver_occ = MBLSE(se_static[1][k], np.array(se_moments_hole[1][k]), log=nlog)
            solver_occ.kernel()

            solver_vir = MBLSE(se_static[1][k], np.array(se_moments_part[1][k]), log=nlog)
            solver_vir.kernel()

            solver = MixedMBLSE(solver_occ, solver_vir)
            e_aux, v_aux = solver.get_auxiliaries()
            se[1].append(SelfEnergy(e_aux, v_aux))

            for s, spin in enumerate(["α", "β"]):
                logger.debug(
                    self,
                    "Error in moments (%s) [kpt %d]: occ = %.6g  vir = %.6g",
                    spin,
                    k,
                    *self.moment_error(se_moments_hole[s][k], se_moments_part[s][k], se[s][k]),
                )

            gf[0].append(se[0][k].get_greens_function(se_static[0][k]))
            gf[1].append(se[1][k].get_greens_function(se_static[1][k]))

        if self.optimise_chempot:
            se[0], opt = minimize_chempot(se[0], se_static[0], sum(self.nocc[0]), occupancy=1)
            se[1], opt = minimize_chempot(se[1], se_static[1], sum(self.nocc[1]), occupancy=1)

        if self.fock_loop:
            gf, se, conv = fock_loop(self, gf, se, integrals=integrals, **self.fock_opts)

        cpt_α, error_α = search_chempot(
            [g.energy for g in gf[0]],
            [g.coupling for g in gf[0]],
            self.nmo[0],
            sum(self.nocc[0]),
            occupancy=1,
        )
        cpt_β, error_β = search_chempot(
            [g.energy for g in gf[1]],
            [g.coupling for g in gf[1]],
            self.nmo[1],
            sum(self.nocc[1]),
            occupancy=1,
        )
        cpt = (cpt_α, cpt_β)
        error = (error_α, error_β)

        for s, spin in enumerate(["α", "β"]):
            for k in self.kpts.loop(1):
                se[s][k].chempot = cpt[s]
                gf[s][k].chempot = cpt[s]
                logger.info(
                    self, "Error in number of electrons (%s) [kpt %d]: %.5g", spin, k, error[s]
                )

        # Calculate energies
        e_1b = self.energy_hf(gf=gf, integrals=integrals) + self.energy_nuc()
        e_2b_g0 = self.energy_gm(se=se, g0=True)
        logger.info(self, "Energies:")
        logger.info(self, "  One-body (G0):         %15.10g", self._scf.e_tot)
        logger.info(self, "  One-body (G):          %15.10g", e_1b)
        logger.info(self, "  Galitskii-Migdal (G0): %15.10g", e_2b_g0)
        if not self.polarizability.lower().startswith("thc"):
            # This is N^4
            e_2b = self.energy_gm(gf=gf, se=se, g0=False)
            logger.info(self, "  Galitskii-Migdal (G):  %15.10g", e_2b)

        return gf, se

    def make_rdm1(self, gf=None):
        """Get the first-order reduced density matrix.

        Parameters
        ----------
        gf : tuple of tuple of GreensFunction, optional
            Green's function at each k-point for each spin channel. If
            `None`, use either `self.gf`, or the mean-field Green's
            function. Default value is `None`.

        Returns
        -------
        rdm1 : numpy.ndarray
            First-order reduced density matrix at each k-point for each
            spin channel.
        """

        if gf is None:
            gf = self.gf
        if gf is None:
            gf = [
                [GreensFunction(e, np.eye(n)) for e in mo_energy]
                for mo_energy, n in zip(self.mo_energy, self.nmo)
            ]

        return np.array([[g.make_rdm1() for g in gs] for gs in gf])

    def energy_hf(self, gf=None, integrals=None):
        """Calculate the one-body (Hartree--Fock) energy.

        Parameters
        ----------
        gf : tuple of GreensFunction, optional
            Green's function at each k-point for each spin channel. If
            `None`, use either `self.gf`, or the mean-field Green's
            function. Default value is `None`.
        integrals : KUIntegrals, optional
            Integrals object. If `None`, generate from scratch. Default
            value is `None`.

        Returns
        -------
        e_1b : float
            One-body energy.
        """

        if gf is None:
            gf = self.gf
        if integrals is None:
            integrals = self.ao2mo()

        h1e = lib.einsum(
            "kpq,skpi,skqj->skij", self._scf.get_hcore(), self.mo_coeff.conj(), self.mo_coeff
        )
        rdm1 = self.make_rdm1()
        fock = integrals.get_fock(rdm1, h1e)

        e_1b = sum(
            energy.hartree_fock(rdm1[0][k], fock[0][k], h1e[0][k]) for k in self.kpts.loop(1)
        )
        e_1b += sum(
            energy.hartree_fock(rdm1[1][k], fock[1][k], h1e[1][k]) for k in self.kpts.loop(1)
        )
        e_1b /= self.nkpts

        return e_1b.real

    def energy_gm(self, gf=None, se=None, g0=True):
        """Calculate the two-body (Galitskii--Migdal) energy.

        Parameters
        ----------
        gf : tuple of tuple of GreensFunction, optional
            Green's function at each k-point for each spin channel. If
            `None`, use `self.gf`. Default value is `None`.
        se : tuple of tuple of SelfEnergy, optional
            Self-energy at each k-point for each spin channel. If
            `None`, use `self.se`. Default value is `None`.
        g0 : bool, optional
            If `True`, use the mean-field Green's function. Default
            value is `True`.

        Returns
        -------
        e_2b : float
            Two-body energy.

        Notes
        -----
        With `g0=False`, this function scales as
        :math:`\mathcal{O}(N^4)` with system size, whereas with
        `g0=True`, it scales as :math:`\mathcal{O}(N^3)`.
        """

        if gf is None:
            gf = self.gf
        if se is None:
            se = self.se

        if g0:
            e_2b_α = sum(
                energy.galitskii_migdal_g0(self.mo_energy[0][k], self.mo_occ[0][k], se[0][k])
                for k in self.kpts.loop(1)
            )
            e_2b_β = sum(
                energy.galitskii_migdal_g0(self.mo_energy[1][k], self.mo_occ[1][k], se[1][k])
                for k in self.kpts.loop(1)
            )
        else:
            e_2b_α = sum(energy.galitskii_migdal(gf[0][k], se[0][k]) for k in self.kpts.loop(1))
            e_2b_β = sum(energy.galitskii_migdal(gf[1][k], se[1][k]) for k in self.kpts.loop(1))

        e_2b = (e_2b_α + e_2b_β) / 2

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
        raise NotImplementedError

    def init_gf(self, mo_energy=None):
        """Initialise the mean-field Green's function.

        Parameters
        ----------
        mo_energy : numpy.ndarray, optional
            Molecular orbital energies at each k-point for each spin
            channel. Default value is `self.mo_energy`.

        Returns
        -------
        gf : tuple of tuple of GreensFunction
            Mean-field Green's function at each k-point for each spin
            channel.
        """

        if mo_energy is None:
            mo_energy = self.mo_energy

        gf = [[], []]
        for s in range(2):
            for k in self.kpts.loop(1):
                gf[s].append(GreensFunction(mo_energy[s][k], np.eye(self.nmo[s])))

            ws = [g.energy for g in gf[s]]
            vs = [g.coupling for g in gf[s]]
            chempot = search_chempot_unconstrained(
                ws, vs, self.nmo[s], sum(self.nocc[s]), occupancy=1
            )[0]

            for k in self.kpts.loop(1):
                gf[s][k].chempot = chempot

            gf[s] = tuple(gf[s])

        return tuple(gf)
