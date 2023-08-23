"""
Spin-restricted one-shot GW via self-energy moment constraints for
molecular systems.
"""

import functools
from collections import defaultdict
from types import MethodType

import numpy as np
from dyson import MBLSE, MixedMBLSE, NullLogger
from pyscf import lib, scf
from pyscf.agf2 import GreensFunction, SelfEnergy, chempot, mpi_helper
from pyscf.agf2.dfragf2 import DFRAGF2
from pyscf.ao2mo import _ao2mo
from pyscf.lib import logger

from momentGW import energy, thc, util
from momentGW.base import BaseGW
from momentGW.fock import fock_loop
from momentGW.ints import Integrals
from momentGW.rpa import RPA
from momentGW.tda import TDA


def kernel(
    gw,
    nmom_max,
    mo_energy,
    mo_coeff,
    moments=None,
    integrals=None,
):
    """Moment-constrained one-shot GW.

    Parameters
    ----------
    gw : BaseGW
        GW object.
    nmom_max : int
        Maximum moment number to calculate.
    mo_energy : numpy.ndarray
        Molecular orbital energies.
    mo_coeff : numpy.ndarray
        Molecular orbital coefficients.
    moments : tuple of numpy.ndarray, optional
        Tuple of (hole, particle) moments, if passed then they will
        be used instead of calculating them. Default value is `None`.
    integrals : Integrals, optional
        Integrals object. If `None`, generate from scratch. Default
        value is `None`.

    Returns
    -------
    conv : bool
        Convergence flag. Always `True` for GW, returned for
        compatibility with other GW methods.
    gf : pyscf.agf2.GreensFunction
        Green's function object
    se : pyscf.agf2.SelfEnergy
        Self-energy object
    qp_energy : numpy.ndarray
        Quasiparticle energies. Always `None` for GW, returned for
        compatibility with other GW methods.
    """

    if integrals is None:
        integrals = gw.ao2mo()

    # Get the static part of the SE
    se_static = gw.build_se_static(
        integrals,
        mo_energy=mo_energy,
        mo_coeff=mo_coeff,
    )

    # Get the moments of the SE
    if moments is None:
        th, tp = gw.build_se_moments(
            nmom_max,
            integrals,
            mo_energy=mo_energy,
        )
    else:
        th, tp = moments

    # Solve the Dyson equation
    gf, se = gw.solve_dyson(th, tp, se_static, integrals=integrals)
    conv = True

    return conv, gf, se, None


class GW(BaseGW):
    __doc__ = BaseGW.__doc__.format(
        description="Spin-restricted one-shot GW via self-energy moment constraints for molecules.",
        extra_parameters="",
    )

    @property
    def name(self):
        return "G0W0"

    _kernel = kernel

    def build_se_static(self, integrals, mo_coeff=None, mo_energy=None):
        """Build the static part of the self-energy, including the
        Fock matrix.

        Parameters
        ----------
        integrals : Integrals
            Integrals object.
        mo_energy : numpy.ndarray, optional
            Molecular orbital energies. Default value is that of
            `self.mo_energy`.
        mo_coeff : numpy.ndarray
            Molecular orbital coefficients. Default value is that of
            `self.mo_coeff`.

        Returns
        -------
        se_static : numpy.ndarray
            Static part of the self-energy. If `self.diagonal_se`,
            non-diagonal elements are set to zero.
        """

        if mo_coeff is None:
            mo_coeff = self.mo_coeff
        if mo_energy is None:
            mo_energy = self.mo_energy

        if getattr(self._scf, "xc", "hf") == "hf":
            se_static = np.zeros_like(self._scf.make_rdm1(mo_coeff=mo_coeff))
        else:
            with util.SilentSCF(self._scf):
                vmf = self._scf.get_j() - self._scf.get_veff()
                dm = self._scf.make_rdm1(mo_coeff=mo_coeff)
                vk = integrals.get_k(dm, basis="ao")

            se_static = vmf - vk * 0.5
            se_static = lib.einsum("...pq,...pi,...qj->...ij", se_static, mo_coeff, mo_coeff)

        if self.diagonal_se:
            se_static = lib.einsum("...pq,pq->...pq", se_static, np.eye(se_static.shape[-1]))

        se_static += lib.einsum("...p,...pq->...pq", mo_energy, np.eye(se_static.shape[-1]))

        return se_static

    def build_se_moments(self, nmom_max, integrals, **kwargs):
        """Build the moments of the self-energy.

        Parameters
        ----------
        nmom_max : int
            Maximum moment number to calculate.
        integrals : Integrals
            Integrals object.

        See functions in `momentGW.rpa` for `kwargs` options.

        Returns
        -------
        se_moments_hole : numpy.ndarray
            Moments of the hole self-energy. If `self.diagonal_se`,
            non-diagonal elements are set to zero.
        se_moments_part : numpy.ndarray
            Moments of the particle self-energy. If `self.diagonal_se`,
            non-diagonal elements are set to zero.
        """

        if self.polarizability == "drpa":
            rpa = RPA(self, nmom_max, integrals, **kwargs)
            return rpa.kernel()

        elif self.polarizability == "drpa-exact":
            rpa = RPA(self, nmom_max, integrals, **kwargs)
            return rpa.kernel(exact=True)

        elif self.polarizability == "dtda":
            tda = TDA(self, nmom_max, integrals, **kwargs)
            return tda.kernel()

        elif self.polarizability == "thc-dtda":
            tda = thc.TDA(self, nmom_max, integrals, **kwargs)
            return tda.kernel()

        else:
            raise NotImplementedError

    def ao2mo(self, transform=True):
        """Get the integrals object.

        Parameters
        ----------
        transform : bool, optional
            Whether to transform the integrals object.

        Returns
        -------
        integrals : Integrals
            Integrals object.
        """

        if self.polarizability.startswith("thc"):
            cls = thc.Integrals
            kwargs = self.thc_opts
        else:
            cls = Integrals
            kwargs = dict(
                compression=self.compression,
                compression_tol=self.compression_tol,
                store_full=self.fock_loop,
            )

        integrals = cls(
            self.with_df,
            self.mo_coeff,
            self.mo_occ,
            **kwargs,
        )
        if transform:
            integrals.transform()

        return integrals

    def solve_dyson(self, se_moments_hole, se_moments_part, se_static, integrals=None):
        """
        Solve the Dyson equation due to a self-energy resulting from a
        list of hole and particle moments, along with a static
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
            Moments of the hole self-energy.
        se_moments_part : numpy.ndarray
            Moments of the particle self-energy.
        se_static : numpy.ndarray
            Static part of the self-energy.
        integrals : Integrals
            Density-fitted integrals.Required if `self.fock_loop` is
            `True`. Default value is `None`.

        Returns
        -------
        gf : pyscf.agf2.GreensFunction
            Green's function.
        se : pyscf.agf2.SelfEnergy
            Self-energy.
        """

        nlog = NullLogger()

        solver_occ = MBLSE(se_static, np.array(se_moments_hole), log=nlog)
        solver_occ.kernel()

        solver_vir = MBLSE(se_static, np.array(se_moments_part), log=nlog)
        solver_vir.kernel()

        solver = MixedMBLSE(solver_occ, solver_vir)
        e_aux, v_aux = solver.get_auxiliaries()
        se = SelfEnergy(e_aux, v_aux)

        if self.optimise_chempot:
            se, opt = chempot.minimize_chempot(se, se_static, self.nocc * 2)

        logger.debug(
            self,
            "Error in moments: occ = %.6g  vir = %.6g",
            *self.moment_error(se_moments_hole, se_moments_part, se),
        )

        gf = se.get_greens_function(se_static)
        gf.energy = mpi_helper.bcast(gf.energy, root=0)
        gf.coupling = mpi_helper.bcast(gf.coupling, root=0)

        if self.fock_loop:
            # TODO remove these try...except
            try:
                gf, se, conv = fock_loop(self, gf, se, integrals=integrals, **self.fock_opts)
            except IndexError:
                pass

        try:
            cpt, error = chempot.binsearch_chempot(
                (gf.energy, gf.coupling),
                gf.nphys,
                self.nocc * 2,
            )
        except IndexError:
            cpt = gf.chempot
            error = np.trace(gf.make_rdm1()) - self.nocc * 2

        se.chempot = cpt
        gf.chempot = cpt
        logger.info(self, "Error in number of electrons: %.5g", error)

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
        """Get the first-order reduced density matrix.

        Parameters
        ----------
        gf : GreensFunction, optional
            Green's function. If `None`, use either `self.gf`, or the
            mean-field Green's function. Default value is `None`.

        Returns
        -------
        rdm1 : numpy.ndarray
            First-order reduced density matrix.
        """

        if gf is None:
            gf = self.gf
        if gf is None:
            gf = self.init_gf()

        return gf.make_rdm1()

    def moment_error(self, se_moments_hole, se_moments_part, se):
        """Return the error in the moments.

        Parameters
        ----------
        se_moments_hole : numpy.ndarray
            Moments of the hole self-energy.
        se_moments_part : numpy.ndarray
            Moments of the particle self-energy.
        se : SelfEnergy
            Self-energy.

        Returns
        -------
        eh : float
            Error in the hole moments.
        ep : float
            Error in the particle moments.
        """

        eh = self._moment_error(
            se_moments_hole,
            se.get_occupied().moment(range(len(se_moments_hole))),
        )
        ep = self._moment_error(
            se_moments_part,
            se.get_virtual().moment(range(len(se_moments_part))),
        )

        return eh, ep

    def energy_nuc(self):
        """Calculate the nuclear repulsion energy."""
        return self._scf.energy_nuc()

    def energy_hf(self, gf=None, integrals=None):
        """Calculate the one-body (Hartree--Fock) energy.

        Parameters
        ----------
        gf : GreensFunction, optional
            Green's function. If `None`, use either `self.gf`, or the
            mean-field Green's function. Default value is `None`.
        integrals : Integrals, optional
            Integrals. If `None`, generate from scratch. Default value
            is `None`.

        Returns
        -------
        e_1b : float
            One-body energy.
        """

        if gf is None:
            gf = self.gf
        if integrals is None:
            integrals = self.ao2mo()

        h1e = np.linalg.multi_dot((self.mo_coeff.T, self._scf.get_hcore(), self.mo_coeff))
        rdm1 = self.make_rdm1()
        fock = integrals.get_fock(rdm1, h1e)

        return energy.hartree_fock(rdm1, fock, h1e)

    def energy_gm(self, gf=None, se=None, g0=True):
        """Calculate the two-body (Galitskii--Migdal) energy.

        Parameters
        ----------
        gf : GreensFunction, optional
            Green's function. If `None`, use `self.gf`. Default value
            is `None`.
        se : SelfEnergy, optional
            Self-energy. If `None`, use `self.se`. Default value is
            `None`.
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
            e_2b = energy.galitskii_migdal_g0(self.mo_energy, self.mo_occ, se)
        else:
            e_2b = energy.galitskii_migdal(gf, se)

        # Extra factor for non-self-consistent G
        e_2b *= 0.5

        return e_2b

    def init_gf(self, mo_energy=None):
        """Initialise the mean-field Green's function.

        Parameters
        ----------
        mo_energy : numpy.ndarray, optional
            Molecular orbital energies. Default value is
            `self.mo_energy`.

        Returns
        -------
        gf : GreensFunction
            Mean-field Green's function.
        """

        if mo_energy is None:
            mo_energy = self.mo_energy

        chempot = 0.5 * (mo_energy[self.nocc - 1] + mo_energy[self.nocc])
        gf = GreensFunction(mo_energy, np.eye(self.nmo), chempot=chempot)

        return gf
