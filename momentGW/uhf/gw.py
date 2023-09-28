"""
Spin-unrestricted one-shot GW via self-energy moment constraints for
molecular systems.
"""

import numpy as np
from dyson import MBLSE, Lehmann, MixedMBLSE, NullLogger
from pyscf import lib
from pyscf.lib import logger

from momentGW import energy, mpi_helper
from momentGW.base import BaseGW
from momentGW.fock import minimize_chempot, search_chempot
from momentGW.gw import GW
from momentGW.uhf.base import BaseUGW
from momentGW.uhf.fock import fock_loop
from momentGW.uhf.ints import UIntegrals
from momentGW.uhf.rpa import dRPA
from momentGW.uhf.tda import dTDA


class UGW(BaseUGW, GW):  # noqa: D101
    __doc__ = BaseGW.__doc__.format(
        description="Spin-unrestricted one-shot GW via self-energy moment constraints for "
        + "molecules.",
        extra_parameters="",
    )

    @property
    def name(self):
        """Method name."""
        polarizability = self.polarizability.upper().replace("DTDA", "dTDA").replace("DRPA", "dRPA")
        return f"{polarizability}-UG0W0"

    def ao2mo(self, transform=True):
        """Get the integrals object.

        Parameters
        ----------
        transform : bool, optional
            Whether to transform the integrals object.

        Returns
        -------
        integrals : UIntegrals
            Integrals object.
        """

        integrals = UIntegrals(
            self.with_df,
            self.mo_coeff,
            self.mo_occ,
            compression=self.compression,
            compression_tol=self.compression_tol,
            store_full=self.fock_loop,
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
        integrals : UIntegrals
            Integrals object.

        See functions in `momentGW.uhf.rpa` for `kwargs` options.

        Returns
        -------
        se_moments_hole : tuple of numpy.ndarray
            Moments of the hole self-energy for each spin channel. If
            `self.diagonal_se`, non-diagonal elements are set to zero.
        se_moments_part : tuple of numpy.ndarray
            Moments of the particle self-energy for each spin channel.
            If `self.diagonal_se`, non-diagonal elements are set to
            zero.
        """

        if self.polarizability.lower() == "dtda":
            tda = dTDA(self, nmom_max, integrals, **kwargs)
            return tda.kernel()

        elif self.polarizability.lower() == "drpa":
            rpa = dRPA(self, nmom_max, integrals, **kwargs)
            return rpa.kernel()

        else:
            raise NotImplementedError

    def solve_dyson(self, se_moments_hole, se_moments_part, se_static, integrals=None):
        """
        Solve the Dyson equation due to a self-energy resulting from a
        list of hole and particle moments, along with a static
        contribution for each spin channel.

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
        se_moments_hole : tuple of numpy.ndarray
            Moments of the hole self-energy for each spin channel.
        se_moments_part : tuple of numpy.ndarray
            Moments of the particle self-energy for each spin channel.
        se_static : tuple of numpy.ndarray
            Static part of the self-energy for each spin channel.
        integrals : UIntegrals
            Integrals object. Required if `self.fock_loop` is `True`.
            Default value is `None`.

        Returns
        -------
        gf : tuple of dyson.Lehmann
            Green's function for each spin channel.
        se : tuple of dyson.Lehmann
            Self-energy for each spin channel.
        """

        nlog = NullLogger()

        solver_occ = MBLSE(se_static[0], np.array(se_moments_hole[0]), log=nlog)
        solver_occ.kernel()

        solver_vir = MBLSE(se_static[0], np.array(se_moments_part[0]), log=nlog)
        solver_vir.kernel()

        solver = MixedMBLSE(solver_occ, solver_vir)
        se_α = solver.get_self_energy()

        solver_occ = MBLSE(se_static[1], np.array(se_moments_hole[1]), log=nlog)
        solver_occ.kernel()

        solver_vir = MBLSE(se_static[1], np.array(se_moments_part[1]), log=nlog)
        solver_vir.kernel()

        solver = MixedMBLSE(solver_occ, solver_vir)
        se_β = solver.get_self_energy()

        se = (se_α, se_β)

        if self.optimise_chempot:
            se_α, opt = minimize_chempot(se[0], se_static[0], self.nocc[0], occupancy=1)
            se_β, opt = minimize_chempot(se[1], se_static[1], self.nocc[1], occupancy=1)
            se = (se_α, se_β)

        logger.debug(
            self,
            "Error in moments (α): occ = %.6g  vir = %.6g",
            *self.moment_error(se_moments_hole[0], se_moments_part[0], se[0]),
        )
        logger.debug(
            self,
            "Error in moments (β): occ = %.6g  vir = %.6g",
            *self.moment_error(se_moments_hole[1], se_moments_part[1], se[1]),
        )

        gf = tuple(
            Lehmann(*s.diagonalise_matrix_with_projection(s_static))
            for s, s_static in zip(se, se_static)
        )
        for g, s in zip(gf, se):
            g.energies = mpi_helper.bcast(g.energies, root=0)
            g.couplings = mpi_helper.bcast(g.couplings, root=0)
            g.chempot = s.chempot

        if self.fock_loop:
            gf, se, conv = fock_loop(self, gf, se, integrals=integrals, **self.fock_opts)

        cpt_α, error_α = search_chempot(
            gf[0].energies,
            gf[0].couplings,
            gf[0].nphys,
            self.nocc[0],
            occupancy=1,
        )
        cpt_β, error_β = search_chempot(
            gf[1].energies,
            gf[1].couplings,
            gf[1].nphys,
            self.nocc[1],
            occupancy=1,
        )
        cpt = (cpt_α, cpt_β)
        error = (error_α, error_β)

        se[0].chempot = cpt[0]
        se[1].chempot = cpt[1]
        gf[0].chempot = cpt[0]
        gf[1].chempot = cpt[1]
        logger.info(self, "Error in number of electrons (α): %.5g", error[0])
        logger.info(self, "Error in number of electrons (β): %.5g", error[1])

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
        gf : tuple of dyson.Lehmann, optional
            Green's function for each spin channel. If `None`, use
            either `self.gf`, or the mean-field Green's function.
            Default value is `None`.

        Returns
        -------
        rdm1 : numpy.ndarray
            First-order reduced density matrix for each spin channel.
        """

        if gf is None:
            gf = self.gf
        if gf is None:
            gf = self.init_gf()

        return (gf[0].occupied().moment(0), gf[1].occupied().moment(0))

    def energy_hf(self, gf=None, integrals=None):
        """Calculate the one-body (Hartree--Fock) energy.

        Parameters
        ----------
        gf : tuple of dyson.Lehmann, optional
            Green's function for each spin channel. If `None`, use
            either `self.gf`, or the mean-field Green's function.
            Default value is `None`.
        integrals : UIntegrals, optional
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

        h1e = tuple(
            lib.einsum("pq,pi,qj->ij", self._scf.get_hcore(), c.conj(), c) for c in self.mo_coeff
        )
        rdm1 = self.make_rdm1(gf=gf)
        fock = integrals.get_fock(rdm1, h1e)

        e_1b = energy.hartree_fock(rdm1[0], fock[0], h1e[0])
        e_1b += energy.hartree_fock(rdm1[1], fock[1], h1e[1])

        return e_1b

    def energy_gm(self, gf=None, se=None, g0=True):
        r"""Calculate the two-body (Galitskii--Migdal) energy.

        Parameters
        ----------
        gf : tuple of dyson.Lehmann, optional
            Green's function for each spin channel. If `None`, use
            `self.gf`. Default value is `None`.
        se : tuple of dyson.Lehmann, optional
            Self-energy for each spin channel. If `None`, use `self.se`.
            Default value is `None`.
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
            e_2b_α = energy.galitskii_migdal_g0(self.mo_energy[0], self.mo_occ[0], se[0])
            e_2b_β = energy.galitskii_migdal_g0(self.mo_energy[1], self.mo_occ[1], se[1])
        else:
            e_2b_α = energy.galitskii_migdal(gf[0], se[0])
            e_2b_β = energy.galitskii_migdal(gf[1], se[1])

        e_2b = (e_2b_α + e_2b_β) / 2

        return e_2b

    def init_gf(self, mo_energy=None):
        """Initialise the mean-field Green's function.

        Parameters
        ----------
        mo_energy : tuple of numpy.ndarray, optional
            Molecular orbital energies for each spin channel. Default
            value is `self.mo_energy`.

        Returns
        -------
        gf : tuple of dyson.Lehmann
            Mean-field Green's function for each spin channel.
        """

        if mo_energy is None:
            mo_energy = self.mo_energy

        gf = [
            Lehmann(mo_energy[0], np.eye(self.nmo[0])),
            Lehmann(mo_energy[1], np.eye(self.nmo[1])),
        ]
        gf[0].chempot = search_chempot(
            gf[0].energies, gf[0].couplings, self.nmo[0], self.nocc[0], occupancy=1
        )[0]
        gf[1].chempot = search_chempot(
            gf[1].energies, gf[1].couplings, self.nmo[1], self.nocc[1], occupancy=1
        )[0]

        return tuple(gf)
