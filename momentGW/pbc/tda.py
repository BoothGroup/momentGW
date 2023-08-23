"""
Construct TDA moments with periodic boundary conditions.
"""

import numpy as np
import scipy.special
from pyscf import lib
from pyscf.agf2 import mpi_helper
from pyscf.pbc import dft
from pyscf.pbc.gw.krgw_ac import get_qij

from momentGW.tda import TDA as MolTDA


class TDA(MolTDA):
    """
    Compute the self-energy moments using dTDA and numerical integration

    with periodic boundary conditions.
    Parameters
    ----------
    gw : BaseKGW
        GW object.
    nmom_max : int
        Maximum moment number to calculate.
    Lpx : numpy.ndarray
        Density-fitted ERI tensor, where the first two indices
        enumerate the k-points, the third index is the auxiliary
        basis function index, and the fourth and fifth indices are
        the MO and Green's function orbital indices, respectively.
    integrals : KIntegrals
        Density-fitted integrals.
    mo_energy : numpy.ndarray or tuple of numpy.ndarray, optional
        Molecular orbital energies at each k-point.  If a tuple is passed,
        the first element corresponds to the Green's function basis and
        the second to the screened Coulomb interaction.  Default value is
        that of `gw.mo_energy`.
    mo_occ : numpy.ndarray or tuple of numpy.ndarray, optional
        Molecular orbital occupancies at each k-point.  If a tuple is
        passed, the first element corresponds to the Green's function basis
        and the second to the screened Coulomb interaction.  Default value
        is that of `gw.mo_occ`.
    """

    def __init__(
        self,
        gw,
        nmom_max,
        integrals,
        mo_energy=None,
        mo_occ=None,
    ):
        self.gw = gw
        self.nmom_max = nmom_max
        self.integrals = integrals

        # Get the MO energies for G and W
        if mo_energy is None:
            self.mo_energy_g = self.mo_energy_w = gw.mo_energy
        elif isinstance(mo_energy, tuple):
            self.mo_energy_g, self.mo_energy_w = mo_energy
        else:
            self.mo_energy_g = self.mo_energy_w = mo_energy

        # Get the MO occupancies for G and W
        if mo_occ is None:
            self.mo_occ_g = self.mo_occ_w = gw.mo_occ
        elif isinstance(mo_occ, tuple):
            self.mo_occ_g, self.mo_occ_w = mo_occ
        else:
            self.mo_occ_g = self.mo_occ_w = mo_occ

        # Options and thresholds
        self.report_quadrature_error = True
        if self.gw.compression and "ia" in self.gw.compression.split(","):
            self.compression_tol = gw.compression_tol
        else:
            self.compression_tol = None

    def build_dd_moments(self):
        """Build the moments of the density-density response."""

        cput0 = (lib.logger.process_clock(), lib.logger.perf_counter())
        lib.logger.info(self.gw, "Building density-density moments")
        lib.logger.debug(self.gw, "Memory usage: %.2f GB", self._memory_usage())

        kpts = self.kpts
        moments = np.zeros((self.nkpts, self.nkpts, self.nmom_max + 1), dtype=object)

        # Get the zeroth order moment
        for q in kpts.loop(1):
            for kj in kpts.loop(1, mpi=True):
                kb = kpts.member(kpts.wrap_around(kpts[q] + kpts[kj]))
                moments[q, kb, 0] += self.integrals.Lia[kj, kb] / self.nkpts
        cput1 = lib.logger.timer(self.gw, "zeroth moment", *cput0)

        # Get the higher order moments
        for i in range(1, self.nmom_max + 1):
            for q in kpts.loop(1):
                for kj in kpts.loop(1, mpi=True):
                    kb = kpts.member(kpts.wrap_around(kpts[q] + kpts[kj]))

                    d = lib.direct_sum(
                        "a-i->ia",
                        self.mo_energy_w[kb][self.mo_occ_w[kb] == 0],
                        self.mo_energy_w[kj][self.mo_occ_w[kj] > 0],
                    )
                    moments[q, kb, i] += moments[q, kb, i - 1] * d.ravel()[None]

                tmp = np.zeros((self.naux[q], self.naux[q]), dtype=complex)
                for ki in kpts.loop(1, mpi=True):
                    ka = kpts.member(kpts.wrap_around(kpts[q] + kpts[ki]))

                    tmp += np.dot(moments[q, ka, i - 1], self.integrals.Lia[ki, ka].T.conj())

                tmp = mpi_helper.allreduce(tmp)
                tmp *= 2.0
                tmp /= self.nkpts

                for kj in kpts.loop(1, mpi=True):
                    kb = kpts.member(kpts.wrap_around(kpts[q] + kpts[kj]))

                    moments[q, kb, i] += np.dot(tmp, self.integrals.Lai[kj, kb].conj())

            cput1 = lib.logger.timer(self.gw, "moment %d" % i, *cput1)

        return moments

    def convolve(self, eta):
        """Handle the convolution of the moments of G and W."""

        kpts = self.kpts

        # Setup dependent on diagonal SE
        if self.gw.diagonal_se:
            pqchar = pchar = qchar = "p"
            fproc = lambda x: np.diag(x)
        else:
            pqchar, pchar, qchar = "pq", "p", "q"
            fproc = lambda x: x

        moments_occ = np.zeros((self.nkpts, self.nmom_max + 1, self.nmo, self.nmo), dtype=complex)
        moments_vir = np.zeros((self.nkpts, self.nmom_max + 1, self.nmo, self.nmo), dtype=complex)
        moms = np.arange(self.nmom_max + 1)
        for n in moms:
            fp = scipy.special.binom(n, moms)
            fh = fp * (-1) ** moms
            for q in kpts.loop(1):
                for kp in kpts.loop(1, mpi=True):
                    kx = kpts.member(kpts.wrap_around(kpts[kp] - kpts[q]))
                    subscript = f"t,kt,kt{pqchar}->{pqchar}"

                    eo = np.power.outer(self.mo_energy_g[kx][self.mo_occ_g[kx] > 0], n - moms)
                    to = lib.einsum(subscript, fh, eo, eta[kp, q][self.mo_occ_g[kx] > 0])
                    moments_occ[kp, n] += fproc(to)

                    ev = np.power.outer(self.mo_energy_g[kx][self.mo_occ_g[kx] == 0], n - moms)
                    tv = lib.einsum(subscript, fp, ev, eta[kp, q][self.mo_occ_g[kx] == 0])
                    moments_vir[kp, n] += fproc(tv)

        # Numerical integration can lead to small non-hermiticity
        for n in range(self.nmom_max + 1):
            for k in kpts.loop(1, mpi=True):
                moments_occ[k, n] = 0.5 * (moments_occ[k, n] + moments_occ[k, n].T.conj())
                moments_vir[k, n] = 0.5 * (moments_vir[k, n] + moments_vir[k, n].T.conj())

        moments_occ = mpi_helper.allreduce(moments_occ)
        moments_vir = mpi_helper.allreduce(moments_vir)

        return moments_occ, moments_vir

    def build_se_moments(self, moments_dd):
        """Build the moments of the self-energy via convolution."""

        cput0 = (lib.logger.process_clock(), lib.logger.perf_counter())
        lib.logger.info(self.gw, "Building self-energy moments")
        lib.logger.debug(self.gw, "Memory usage: %.2f GB", self._memory_usage())

        kpts = self.kpts

        # Setup dependent on diagonal SE
        if self.gw.diagonal_se:
            pqchar = pchar = qchar = "p"
            eta_shape = lambda k: (self.mo_energy_g[k].size, self.nmom_max + 1, self.nmo)
        else:
            pqchar, pchar, qchar = "pq", "p", "q"
            eta_shape = lambda k: (self.mo_energy_g[k].size, self.nmom_max + 1, self.nmo, self.nmo)
        eta = np.zeros((self.nkpts, self.nkpts), dtype=object)

        # Get the moments in (aux|aux) and rotate to (mo|mo)
        for n in range(self.nmom_max + 1):
            for q in kpts.loop(1):
                eta_aux = 0
                for kj in kpts.loop(1, mpi=True):
                    kb = kpts.member(kpts.wrap_around(kpts[q] + kpts[kj]))
                    eta_aux += np.dot(moments_dd[q, kb, n], self.integrals.Lia[kj, kb].T.conj())

                eta_aux = mpi_helper.allreduce(eta_aux)
                eta_aux *= 2.0
                eta_aux /= self.nkpts

                for kp in kpts.loop(1, mpi=True):
                    kx = kpts.member(kpts.wrap_around(kpts[kp] - kpts[q]))

                    if not isinstance(eta[kp, q], np.ndarray):
                        eta[kp, q] = np.zeros(eta_shape(kx), dtype=eta_aux.dtype)

                    for x in range(self.mo_energy_g[kx].size):
                        Lp = self.integrals.Lpx[kp, kx][:, :, x]
                        subscript = f"P{pchar},Q{qchar},PQ->{pqchar}"
                        eta[kp, q][x, n] += lib.einsum(subscript, Lp, Lp.conj(), eta_aux)

        cput1 = lib.logger.timer(self.gw, "rotating DD moments", *cput0)

        # Construct the self-energy moments
        moments_occ, moments_vir = self.convolve(eta)
        cput1 = lib.logger.timer(self.gw, "constructing SE moments", *cput1)

        if self.gw.fc:
            moments_dd_fc = self.build_dd_moments_fc()

            moments_occ_fc, moments_vir_fc = self.build_se_moments_fc(*moments_dd_fc)

            moments_occ += moments_occ_fc
            moments_vir += moments_vir_fc

            cput1 = lib.logger.timer(self.gw, "fc correction", *cput1)

        return moments_occ, moments_vir

    def build_dd_moments_exact(self):
        raise NotImplementedError

    def build_dd_moments_fc(self):
        """
        Build the moments of the "head" (G=0, G'=0) and "wing"
        (G=P, G'=0) density-density response.
        """

        kpts = self.kpts
        integrals = self.integrals

        # q-point for q->0 finite size correction
        qpt = np.array([1e-3, 0.0, 0.0])
        qpt = self.kpts.get_abs_kpts(qpt)

        # 1/sqrt(Ω) * ⟨Ψ_{ik}|e^{iqr}|Ψ_{ak-q}⟩
        qij = self.build_pert_term(qpt)

        # Build the DD moments for the "head" (G=0, G'=0) correction
        moments_head = np.zeros((self.nkpts, self.nmom_max + 1), dtype=complex)
        for k in kpts.loop(1):
            d = lib.direct_sum(
                "a-i->ia",
                self.mo_energy_w[k][self.mo_occ_w[k] == 0],
                self.mo_energy_w[k][self.mo_occ_w[k] > 0],
            )
            dn = np.ones_like(d)
            for n in range(self.nmom_max + 1):
                moments_head[k, n] = lib.einsum("ia,ia,ia->", qij[k], qij[k].conj(), dn)
                dn *= d

        # Build the DD moments for the "wing" (G=P, G'=0) correction
        moments_wing = np.zeros((self.nkpts, self.nmom_max + 1), dtype=object)
        for k in kpts.loop(1):
            d = lib.direct_sum(
                "a-i->ia",
                self.mo_energy_w[k][self.mo_occ_w[k] == 0],
                self.mo_energy_w[k][self.mo_occ_w[k] > 0],
            )
            dn = np.ones_like(d)
            for n in range(self.nmom_max + 1):
                moments_wing[k, n] = lib.einsum(
                    "Lx,x,x->L",
                    integrals.Lia[k, k],
                    qij[k].conj().ravel(),
                    dn.ravel(),
                )
                dn *= d

        moments_head *= -4.0 * np.pi
        moments_wing *= -4.0 * np.pi

        return moments_head, moments_wing

    def build_se_moments_fc(self, moments_head, moments_wing):
        """
        Build the moments of the self-energy corresponding to the
        "wing" (G=P, G'=0) and "head" (G=0, G'=0) density-density
        response via convolution.
        """

        kpts = self.kpts
        integrals = self.integrals
        moms = np.arange(self.nmom_max + 1)

        # Construct the self-energy moments for the "head" (G=0, G'=0)
        moments_occ_h = np.zeros((self.nkpts, self.nmom_max + 1, self.nmo, self.nmo), dtype=complex)
        moments_vir_h = np.zeros((self.nkpts, self.nmom_max + 1, self.nmo, self.nmo), dtype=complex)
        for n in moms:
            fp = scipy.special.binom(n, moms)
            fh = fp * (-1) ** moms
            for k in kpts.loop(1):
                eo = np.power.outer(self.mo_energy_g[k][self.mo_occ_g[k] > 0], n - moms)
                to = lib.einsum("t,kt,t->", fh, eo, moments_head[k])
                moments_occ_h[k, n] = np.diag([to] * self.nmo)

                ev = np.power.outer(self.mo_energy_g[k][self.mo_occ_g[k] == 0], n - moms)
                tv = lib.einsum("t,kt,t->", fp, ev, moments_head[k])
                moments_vir_h[k, n] = np.diag([tv] * self.nmo)

        # Construct the self-energy moments for the "wing" (G=P, G'=0)
        moments_occ_w = np.zeros((self.nkpts, self.nmom_max + 1, self.nmo, self.nmo), dtype=complex)
        moments_vir_w = np.zeros((self.nkpts, self.nmom_max + 1, self.nmo, self.nmo), dtype=complex)
        for n in moms:
            fp = scipy.special.binom(n, moms)
            fh = fp * (-1) ** moms
            for k in kpts.loop(1):
                eta = np.zeros(
                    (self.integrals.nmo_g[k], self.nmom_max + 1, self.nmo), dtype=complex
                )
                for t in moms:
                    eta[:, t] = lib.einsum("L,Lpx->xp", moments_wing[k, t], integrals.Lpx[k, k])

                eo = np.power.outer(self.mo_energy_g[k][self.mo_occ_g[k] > 0], n - moms)
                to = lib.einsum("t,kt,ktp->p", fh, eo, eta[self.mo_occ_g[k] > 0])
                moments_occ_w[k, n] = np.diag(to)

                ev = np.power.outer(self.mo_energy_g[k][self.mo_occ_g[k] == 0], n - moms)
                tv = lib.einsum("t,kt,ktp->p", fp, ev, eta[self.mo_occ_g[k] == 0])
                moments_vir_w[k, n] = np.diag(tv)

        moments_occ = moments_occ_h + moments_occ_w
        moments_vir = moments_vir_h + moments_vir_w

        factor = -2.0 / np.pi * (6.0 * np.pi**2 / self.gw.cell.vol / self.nkpts) ** (1.0 / 3.0)
        moments_occ *= factor
        moments_vir *= factor

        return moments_occ, moments_vir

    def build_pert_term(self, qpt):
        """
        Compute 1/sqrt(Ω) * ⟨Ψ_{ik}|e^{iqr}|Ψ_{ak-q}⟩ at q-point index
        q using perturbation theory.
        """

        coords, weights = dft.gen_grid.get_becke_grids(self.gw.cell, level=5)
        ngrid = len(coords)

        qij = np.zeros((self.nkpts,), dtype=object)
        for k in self.kpts.loop(1):
            ao_p = dft.numint.eval_ao(self.gw.cell, coords, kpt=self.kpts[k], deriv=1)
            ao, ao_grad = ao_p[0], ao_p[1:4]

            ao_ao_grad = lib.einsum("g,gm,xgn->xmn", weights, ao.conj(), ao_grad)
            q_ao_ao_grad = lib.einsum("x,xmn->mn", qpt, ao_ao_grad) * -1.0j
            q_mo_mo_grad = lib.einsum(
                "mn,mi,na->ia",
                q_ao_ao_grad,
                self.integrals.mo_coeff_w[k][:, self.mo_occ_w[k] > 0].conj(),
                self.integrals.mo_coeff_w[k][:, self.mo_occ_w[k] == 0],
            )

            d = lib.direct_sum(
                "a-i->ia",
                self.mo_energy_w[k][self.mo_occ_w[k] == 0],
                self.mo_energy_w[k][self.mo_occ_w[k] > 0],
            )
            dens = q_mo_mo_grad / d
            qij[k] = dens / np.sqrt(self.gw.cell.vol)

        return qij

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
