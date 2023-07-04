"""
Spin-restricted one-shot GW via self-energy moment constraints for
molecular systems.
"""

from types import MethodType

import h5py
import time
import numpy as np
from dyson import MBLSE, MixedMBLSE, NullLogger
from pyscf import lib, scf
from pyscf.agf2 import GreensFunction, SelfEnergy, chempot, mpi_helper
from pyscf.agf2.dfragf2 import DFRAGF2
from pyscf.ao2mo import _ao2mo
from pyscf.lib import logger

from momentGW import rpa
from momentGW.base import BaseGW



def kernel(
    gw,
    nmom_max,
    ppoints,
    mo_energy,
    mo_coeff,
    calc_type,
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
    integrals : tuple of numpy.ndarray, optional
        Density-fitted ERI tensors. If None, generate from `gw.ao2mo`.
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
        integrals = gw.ao2mo(mo_coeff)
    Lpq, Lia = integrals

    # Get the static part of the SE
    se_static = gw.build_se_static(
        Lpq=Lpq,
        mo_energy=mo_energy,
        mo_coeff=mo_coeff,
    )

    # Get the moments of the SE
    if moments is None:
        th, tp = gw.build_se_moments(
            nmom_max,
            ppoints,
            Lpq,
            Lia,
            calc_type,
            mo_energy=mo_energy,
        )
    else:
        th, tp = moments

    # Solve the Dyson equation
    gf, se, errors = gw.solve_dyson(th, tp, se_static, Lpq=Lpq)
    conv = True

    return conv, gf, se, errors





class GW(BaseGW):
    __doc__ = BaseGW.__doc__.format(
        description="Spin-restricted one-shot GW via self-energy moment constraints for molecules.",
        extra_parameters="",
    )

    @property
    def name(self):
        return "G0W0"

    def build_se_static(self, Lpq=None, mo_coeff=None, mo_energy=None):
        """Build the static part of the self-energy, including the
        Fock matrix.

        Parameters
        ----------
        Lpq : np.ndarray, optional
            Density-fitted ERI tensor. If None, generate from `gw.ao2mo`.
            Default value is None.
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
        if Lpq is None and self.vhf_df:
            Lpq, _ = self.ao2mo(mo_coeff)
        with lib.temporary_env(self._scf, verbose=10):
            with lib.temporary_env(self._scf.with_df, verbose=10):
                v_mf = self._scf.get_veff() - self._scf.get_j()
                dm = self._scf.make_rdm1(mo_coeff=mo_coeff)
        # v_mf = v_mf[0]
        # mo_coeff = mo_coeff[0]
        v_mf = lib.einsum("pq,pi,qj->ij", v_mf, mo_coeff, mo_coeff)

        # v_hf from DFT/HF density
        if self.vhf_df:
            vk = np.zeros_like(v_mf)
            p0, p1 = list(mpi_helper.prange(0, self.nmo, self.nmo))[0]

            sc = np.dot(self._scf.get_ovlp(), mo_coeff)
            dm = lib.einsum("pq,pi,qj->ij", dm, sc, sc)

            tmp = lib.einsum("Qik,kl->Qil", Lpq, dm[p0:p1])
            tmp = mpi_helper.allreduce(tmp)
            vk[:, p0:p1] = -lib.einsum("Qil,Qlj->ij", tmp, Lpq) * 0.5
            vk = mpi_helper.allreduce(vk)
        else:
            with lib.temporary_env(self._scf.with_df, verbose=0):
                with lib.temporary_env(self._scf.with_df, verbose=0):
                    vk = scf.hf.SCF.get_veff(self._scf, self.mol, dm)
                    vk -= scf.hf.SCF.get_j(self._scf, self.mol, dm)
            # vk = vk[0]
            vk = lib.einsum("pq,pi,qj->ij", vk, mo_coeff, mo_coeff)

        se_static = vk - v_mf

        if self.diagonal_se:
            se_static = np.diag(np.diag(se_static))

        se_static += np.diag(mo_energy)

        return se_static

    def ao2mo(self, mo_coeff, mo_coeff_g=None, mo_coeff_w=None, nocc_w=None):
        """
        Get the density-fitted integrals. This routine returns two
        arrays, allowing self-consistency in G or W.

        When using MPI, these integral arrays are distributed as
        follows. The `Lia` array is distributed over its second and
        third indices, and the `Lpx` array over its third index. The
        distribution is according to
        `pyscf.agf2.mpi_helper.prange(0, N, N)`.

        Parameters
        ----------
        mo_coeff : numpy.ndarray
            Molecular orbital coefficients.
        mo_coeff_g : numpy.ndarray, optional
            Molecular orbital coefficients corresponding to the
            Green's function. Default value is that of `mo_coeff`.
        mo_coeff_w : numpy.ndarray, optional
            Molecular orbital coefficients corresponding to the
            screened Coulomb interaction. Default value is that of
            `mo_coeff`.
        nocc_w : int, optional
            Number of occupied orbitals corresponding to the
            screened Coulomb interaction. Must be specified if
            `mo_coeff_w` is specified.

        Returns
        -------
        Lpx : numpy.ndarray
            Density-fitted ERI tensor, where the first index is
            the auxiliary basis function index, and the second and
            third indices are the MO and Green's function orbital
            indices, respectively.
        Lia : numpy.ndarray
            Density-fitted ERI tensor, where the first index is
            the auxiliary basis function index, and the second and
            third indices are the occupied and virtual screened
            Coulomb interaction orbital indices, respectively.
        """
        naux = self.with_df.get_naoaux()
        if mo_coeff_g is None:
            mo = np.asarray(mo_coeff, order="F")
            nmo = nqmo = mo.shape[-1]
            ijslice = (0, nmo, 0, nmo)
        else:
            mo = np.asarray(np.concatenate([mo_coeff, mo_coeff_g], axis=1), order="F")
            nmo = mo_coeff.shape[-1]
            nqmo = mo_coeff_g.shape[-1]
            ijslice = (0, nmo, nmo, nmo + nqmo)

        # Loop over the auxiliaries so that the block can be
        # transformed and then the MPI slice can be taken, without
        # having to ever store the full transformed block on any
        # given process.
        # FIXME this doesn't distribute the CPU load
        # TODO calculate block size

        p0, p1 = list(mpi_helper.prange(0, nqmo, nqmo))[0]
        Lpx = np.zeros((naux, nmo, p1-p0))
        for q0, q1 in lib.prange(0, naux, 5000):
            # f = h5py.File('saved_cderi.h5', 'r')
            # self.with_df._cderi = f['j3c'][:]
            Lpx_block = _ao2mo.nr_e2(self.with_df._cderi2[q0:q1], mo, ijslice, aosym="s2", out=None)
            Lpx_block = Lpx_block.reshape(q1-q0, nmo, nqmo)
            Lpx[q0:q1] = Lpx_block[:, :, p0:p1]

        if mo_coeff_g is None and mo_coeff_w is None and mpi_helper.size == 1:
            #print('hi',self.nocc)
            nov = self.nocc * (self.nmo - self.nocc)
            Lia = Lpx[:, :self.nocc, self.nocc:].reshape(naux, -1)
            return Lpx, Lia

        if mo_coeff_w is None:
            mo = np.asarray(mo_coeff, order="F")
            nocc = self.nocc
            nvir = self.nmo - nocc
            ijslice = (0, nocc, nocc, nocc + nvir)
        else:
            assert nocc_w is not None
            mo = np.asarray(mo_coeff_w, order="F")
            nocc = nocc_w
            nvir = mo.shape[-1] - nocc
            ijslice = (0, nocc, nocc, nocc + nvir)

        p0, p1 = list(mpi_helper.prange(0, nocc*nvir, nocc*nvir))[0]
        Lia = np.zeros((naux, p1-p0))
        for q0, q1 in lib.prange(0, naux, 5000):
            Lia_block = _ao2mo.nr_e2(self.with_df._cderi[q0:q1], mo, ijslice, aosym="s2", out=None)
            Lia_block = Lia_block.reshape(q1-q0, nocc*nvir)
            Lia[q0:q1] = Lia_block[:, p0:p1]

        return Lpx, Lia

    def build_se_moments(self, nmom_max, ppoints, Lpq, Lia, calc_type, **kwargs):
        """Build the moments of the self-energy.

        Parameters
        ----------
        nmom_max : int
            Maximum moment number to calculate.
        Lpq : numpy.ndarray
            Density-fitted ERI tensor. See `self.ao2mo` for details.
        Lia : numpy.ndarray
            Density-fitted ERI tensor. See `self.ao2mo` for details.

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
            return rpa.build_se_moments_drpa(
                self,
                nmom_max,
                ppoints,
                Lpq,
                Lia,
                calc_type,
                **kwargs,
            )

        elif self.polarizability == "drpa-exact":
            # Use exact dRPA
            # FIXME for Lpq, Lia changes
            return rpa.build_se_moments_drpa_exact(
                self,
                nmom_max,
                Lpq,
                **kwargs,
            )

    def solve_dyson(self, se_moments_hole, se_moments_part, se_static, Lpq=None):
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
        Lpq : np.ndarray, optional
            Density-fitted ERI tensor.  Required if `self.fock_loop` is
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

        if self.fock_loop:
            if Lpq is None:
                raise ValueError("Lpq must be passed to solve_dyson if fock_loop=True")

            get_jk = MethodType(DFRAGF2.get_jk, self)
            get_fock = MethodType(DFRAGF2.get_fock, self)

            eri = lambda: None
            eri.eri = lib.pack_tril(Lpq, axis=-1)
            eri.h1e = np.linalg.multi_dot((self.mo_coeff.T, self._scf.get_hcore(), self.mo_coeff))
            eri.nmo = self.nmo
            eri.nocc = self.nocc

            with lib.temporary_env(
                self,
                get_jk=get_jk,
                get_fock=get_fock,
                **self.fock_opts,
            ):
                gf, se, conv = DFRAGF2.fock_loop(self, eri, gf, se)

        try:
            cpt, error = chempot.binsearch_chempot(
                (gf.energy, gf.coupling),
                gf.nphys,
                self.nocc * 2,
            )
        except:
            cpt = gf.chempot
            error = np.trace(gf.make_rdm1()) - gw.nocc * 2

        se.chempot = cpt
        gf.chempot = cpt
        logger.info(self, "Error in number of electrons: %.5g", error)

        errors = self.moment_error(se_moments_hole, se_moments_part, se)

        return gf, se, errors

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
            se_moments_hole[0],
            se.get_occupied().moment(range(len(se_moments_hole)))[0],
        )
        ep = self._moment_error(
            se_moments_part[0],
            se.get_virtual().moment(range(len(se_moments_part)))[0],
        )

        return eh, ep

    def kernel(
        self,
        nmom_max,
        ppoints,
        mo_energy=None,
        mo_coeff=None,
        moments=None,
        integrals=None,
        calc_type=None,

    ):
        if mo_coeff is None:
            mo_coeff = self.mo_coeff
        if mo_energy is None:
            mo_energy = self.mo_energy
        if calc_type is None:
            calc_type = "normal"
            print("Calculation type defaulted to normal")
        if not any(calc_type in s for s in ["normal", 'thc', 'asym_linear']):
            raise ValueError("calc_type must be normal, thc or asym_linear")

        cput0 = (logger.process_clock(), logger.perf_counter())
        self.dump_flags()
        logger.info(self, "nmom_max = %d", nmom_max)

        self.converged, self.gf, self.se, errors = kernel(
            self,
            nmom_max,
            ppoints,
            mo_energy,
            mo_coeff,
            calc_type,
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
        return -gf_occ.energy,gf_vir.energy,errors

