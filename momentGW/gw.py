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

from momentGW import util
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
        be used instead of calculating them. Default value is None.
    integrals : Integrals, optional
        Density-fitted integrals. If None, generate from scratch.
        Default value is None.

    Returns
    -------
    conv : bool
        Convergence flag. Always True for AGW, returned for
        compatibility with other GW methods.
    gf : pyscf.agf2.GreensFunction
        Green's function object
    se : pyscf.agf2.SelfEnergy
        Self-energy object
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

    return conv, gf, se


class GW(BaseGW):
    __doc__ = BaseGW.__doc__.format(
        description="Spin-restricted one-shot GW via self-energy moment constraints for molecules.",
        extra_parameters="",
    )

    @property
    def name(self):
        return "G0W0"

    def build_se_static(self, integrals, mo_coeff=None, mo_energy=None):
        """Build the static part of the self-energy, including the
        Fock matrix.

        Parameters
        ----------
        integrals : Integrals
            Density-fitted integrals.
        mo_energy : numpy.ndarray, optional
            Molecular orbital energies.  Default value is that of
            `self.mo_energy`.
        mo_coeff : numpy.ndarray
            Molecular orbital coefficients.  Default value is that of
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
            se_static = np.zeros((self.nmo, self.nmo))
        else:
            with util.SilentSCF(self._scf):
                vmf = self._scf.get_j() - self._scf.get_veff()
                dm = self._scf.make_rdm1(mo_coeff=mo_coeff)
                vk = integrals.get_k(dm, basis="ao")

            se_static = vmf - vk * 0.5
            se_static = lib.einsum("pq,pi,qj->ij", se_static, mo_coeff, mo_coeff)

        if self.diagonal_se:
            se_static = np.diag(np.diag(se_static))

        se_static += np.diag(mo_energy)

        return se_static

    def build_se_moments(self, nmom_max, integrals, **kwargs):
        """Build the moments of the self-energy.

        Parameters
        ----------
        nmom_max : int
            Maximum moment number to calculate.
        integrals : Integrals
            Density-fitted integrals.

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

        else:
            raise NotImplementedError

    def ao2mo(self):
        """Get the integrals."""

        integrals = Integrals(
            self.with_df,
            self.mo_coeff,
            self.mo_occ,
            compression=self.compression,
            compression_tol=self.compression_tol,
            store_full=self.fock_loop,
        )
        integrals.transform()

        return integrals

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
            Moments of the hole self-energy.
        se_moments_part : numpy.ndarray
            Moments of the particle self-energy.
        se_static : numpy.ndarray
            Static part of the self-energy.
        integrals : Integrals
            Density-fitted integrals.  Required if `self.fock_loop` is
            `True`.  Default value is `None`.

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

        return gf, se

    def make_rdm1(self, gf=None):
        """Get the first-order reduced density matrix."""

        if gf is None:
            gf = self.gf
        if gf is None:
            gf = GreensFunction(self.mo_energy, np.eye(self.nmo))

        return gf.make_rdm1()

    def moment_error(self, se_moments_hole, se_moments_part, se):
        """Return the error in the moments."""

        eh = self._moment_error(
            se_moments_hole,
            se.get_occupied().moment(range(len(se_moments_hole))),
        )
        ep = self._moment_error(
            se_moments_part,
            se.get_virtual().moment(range(len(se_moments_part))),
        )

        return eh, ep

    def kernel(
        self,
        nmom_max,
        mo_energy=None,
        mo_coeff=None,
        moments=None,
        integrals=None,
    ):
        if mo_coeff is None:
            mo_coeff = self.mo_coeff
        if mo_energy is None:
            mo_energy = self.mo_energy

        cput0 = (logger.process_clock(), logger.perf_counter())
        self.dump_flags()
        logger.info(self, "nmom_max = %d", nmom_max)

        self.converged, self.gf, self.se = kernel(
            self,
            nmom_max,
            mo_energy,
            mo_coeff,
            integrals=integrals,
        )

        gf_occ = self.gf.get_occupied()
        gf_occ.remove_uncoupled(tol=1e-1)
        for n in range(min(5, gf_occ.naux)):
            en = -gf_occ.energy[-(n + 1)]
            vn = gf_occ.coupling[:, -(n + 1)]
            qpwt = np.linalg.norm(vn) ** 2
            logger.note(self, "IP energy level %d E = %.16g  QP weight = %0.6g", n, en, qpwt)

        gf_vir = self.gf.get_virtual()
        gf_vir.remove_uncoupled(tol=1e-1)
        for n in range(min(5, gf_vir.naux)):
            en = gf_vir.energy[n]
            vn = gf_vir.coupling[:, n]
            qpwt = np.linalg.norm(vn) ** 2
            logger.note(self, "EA energy level %d E = %.16g  QP weight = %0.6g", n, en, qpwt)

        logger.timer(self, self.name, *cput0)

        return self.converged, self.gf, self.se
