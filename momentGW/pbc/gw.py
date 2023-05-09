"""
Spin-restricted one-shot GW via self-energy moment constraints for
periodic systems.
"""

import numpy as np
from dyson import NullLogger, MBLSE, MixedMBLSE
from pyscf import lib
from pyscf.pbc import scf
from pyscf.agf2 import GreensFunction, SelfEnergy, chempot
from pyscf.lib import logger

from momentGW.pbc.base import BaseKGW
from momentGW.gw import kernel


class KGW(BaseKGW):
    __doc__ = BaseKGW.__doc__.format(
        description="Spin-restricted one-shot GW via self-energy moment constraints for " + \
                "periodic systems.",
        extra_parameters="",
    )

    @property
    def name(self):
        return "KG0W0"

    def build_se_static(self, Lpq=None, mo_coeff=None, mo_energy=None):
        """Build the static part of the self-energy, including the
        Fock matrix.

        Parameters
        ----------
        Lpq : np.ndarray, optional
            Density-fitted ERI tensor. If None, generate from `gw.ao2mo`.
            Default value is None.
        mo_energy : numpy.ndarray, optional
            Molecular orbital energies at each k-point.  Default value
            is that of `self.mo_energy`.
        mo_coeff : numpy.ndarray
            Molecular orbital coefficients at each k-point.  Default
            value is that of `self.mo_coeff`.

        Returns
        -------
        se_static : numpy.ndarray
            Static part of the self-energy at each k-point. If
            `self.diagonal_se`, non-diagonal elements are set to zero.
        """

        if mo_coeff is None:
            mo_coeff = self.mo_coeff
        if mo_energy is None:
            mo_energy = self.mo_energy
        if Lpq is None and self.vhf_df:
            Lpq = self.ao2mo(mo_coeff)

        with lib.temporary_env(self._scf, verbose=0):
            with lib.temporary_env(self._scf.with_df, verbose=0):
                dm = np.array(self._scf.make_rdm1(mo_coeff=mo_coeff))
                v_mf = self._scf.get_veff() - self._scf.get_j(dm_kpts=dm)
        v_mf = lib.einsum("kpq,kpi,kqj->kij", v_mf, mo_coeff.conj(), mo_coeff)

        if self.vhf_df:
            raise NotImplementedError  # TODO
        else:
            with lib.temporary_env(self._scf, verbose=0):
                with lib.temporary_env(self._scf.with_df, verbose=0):
                    vk = scf.khf.KSCF.get_veff(self._scf, self.cell, dm)
                    vk -= scf.khf.KSCF.get_j(self._scf, self.cell, dm)
            vk = lib.einsum("pq,pi,qj->ij", vk, mo_coeff, mo_coeff)

        se_static = vk - v_mf

        if self.diagonal_se:
            se_static = se_static[:, np.diag_indices_from(se_static[0])] = 0.0

        se_static += np.array([np.diag(e) for e in mo_energy])

        return se_static

    def ao2mo(self, mo_coeff, mo_coeff_g=None, mo_coeff_w=None, nocc_w=None):
        """
        Get the density-fitted integrals. This routine returns two
        arrays, allowing self-consistency in G or W.

        Parameters
        ----------
        mo_coeff : numpy.ndarray
            Molecular orbital coefficients at each k-point.
        mo_coeff_g : numpy.ndarray, optional
            Molecular orbital coefficients corresponding to the
            Green's function at each k-point.  Default value is that
            of `mo_coeff`.
        mo_coeff_w : numpy.ndarray, optional
            Molecular orbital coefficients corresponding to the
            screened Coulomb interaction at each k-point.  Default
            value is that of `mo_coeff`.
        nocc_w : int, optional
            Number of occupied orbitals corresponding to the
            screened Coulomb interaction at each k-point.  Must be
            specified if `mo_coeff_w` is specified.

        Returns
        -------
        Lpx : numpy.ndarray
            Density-fitted ERI tensor, where the first two indices
            enumerate the k-points, the third index is the auxiliary
            basis function index, and the fourth and fifth indices are
            the MO and Green's function orbital indices, respectively.
        Lia : numpy.ndarray
            Density-fitted ERI tensor, where the first two indices
            enumerate the k-points, the third index is the auxiliary
            basis function index, and the fourth and fifth indices are
            the occupied and virtual screened Coulomb interaction
            orbital indices, respectively.
        """

        if not (mo_coeff_g is None and mo_coeff_w is None and nocc_w is None):
            raise NotImplementedError  # TODO

        Lpq = lib.einsum("xyLpq,xpi,yqj->xyLij", self.with_df._cderi, mo_coeff, mo_coeff)

        # occ-vir blocks may be ragged due to different numbers of
        # occupied orbitals at each k-point
        Lia = np.empty(shape=(self.nkpts, self.nkpts), dtype=object)
        for ki in range(self.nkpts):
            for kj in range(self.nkpts):
                Lia[ki, kj] = Lpq[ki, kj, :, :self.nocc[ki], self.nocc[kj]:]

        return Lpq, Lia

    def build_se_moments(self, nmom_max, Lpq, Lia, **kwargs):
        """Build the moments of the self-energy.

        Parameters
        ----------
        nmom_max : int
            Maximum moment number to calculate.
        Lpq : numpy.ndarray
            Density-fitted ERI tensor at each k-point. See `self.ao2mo` for
            details.
        Lia : numpy.ndarray
            Density-fitted ERI tensor at each k-point. See `self.ao2mo` for
            details.

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

        raise NotImplementedError  # TODO

    def solve_dyson(self):
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
        Lpq : np.ndarray, optional
            Density-fitted ERI tensor at each k-point.  Required if
            `self.fock_loop` is `True`.  Default value is `None`.

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
        for ki in range(self.nkpts):
            solver_occ = MBLSE(se_static[ki], np.array(se_moments_hole[ki]), log=nlog)
            solver_occ.kernel()

            solver_vir = MBLSE(se_static[ki], np.array(se_moments_part[ki]), log=nlog)
            solver_vir.kernel()

            solver = MixedMBLSE(solver_occ, solver_vir)
            e_aux, v_aux = solver.get_auxiliaries()
            se.append(SelfEnergy(e_aux, v_aux))

            if self.optimise_chempot:
                se[ki], opt = chempot.minimize_chempot(se[ki], se_static[ki], self.nocc[ki] * 2)

            logger.debug(
                self,
                "Error in moments [kpt %d]: occ = %.6g  vir = %.6g",
                *self.moment_error(se_moments_hole[ki], se_moments_part[ki], se[ki]),
            )

            gf.append(se.get_greens_function(se_static[ki]))

            if self.fock_loop:
                raise NotImplementedError

            try:
                cpt, error = chempot.binsearch_chempot(
                    (gf[ki].energy, gf[ki].coupling),
                    gf[ki].nphys,
                    self.nocc[ki] * 2,
                )
            except:
                cpt = gf[ki].chempot
                error = np.trace(gf[ki].make_rdm1()) - self.nocc[ki] * 2

            se[ki].chempot = cpt
            gf[ki].chempot = cpt
            logger.info(self, "Error in number of electrons [kpt %d]: %.5g", ki, error)

        return gf, se


    def make_rdm1(self, gf=None):
        """Get the first-order reduced density matrix at each k-point."""

        if gf is None:
            gf = self.gf
        if gf is None:
            gf = [GreensFunction(self.mo_energy, np.eye(self.nmo))]

        return np.array([g.make_rdm1() for g in gf])

    #def moment_error(self, se_moments_hole, se_moments_part, se):
    #    """Return the error in the moments."""

    #    eh = [
    #        self._moment_error(
    #            se_moments_hole[ki],
    #            se[ki].get_occupied().moment(range(len(se_moments_hole[ki]))),
    #        )
    #        for ki in range(self.nkpts)
    #    ]
    #    ep = [
    #        self._moment_error(
    #            se_moments_part[ki],
    #            se[ki].get_virtual().moment(range(len(se_moments_part[ki]))),
    #        )
    #        for ki in range(self.nkpts)
    #    ]

    #    return eh, ep

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

        gf_occ = self.gf[0].get_occupied()
        gf_occ.remove_uncoupled(tol=1e-1)
        for n in range(min(5, gf_occ.naux)):
            en = -gf_occ.energy[-(n + 1)]
            vn = gf_occ.coupling[:, -(n + 1)]
            qpwt = np.linalg.norm(vn) ** 2
            logger.note(self, "IP energy level (Γ) %d E = %.16g  QP weight = %0.6g", n, en, qpwt)

        gf_vir = self.gf[0].get_virtual()
        gf_vir.remove_uncoupled(tol=1e-1)
        for n in range(min(5, gf_vir.naux)):
            en = gf_vir.energy[n]
            vn = gf_vir.coupling[:, n]
            qpwt = np.linalg.norm(vn) ** 2
            logger.note(self, "EA energy level (Γ) %d E = %.16g  QP weight = %0.6g", n, en, qpwt)

        logger.timer(self, self.name, *cput0)

        return self.converged, self.gf, self.se
