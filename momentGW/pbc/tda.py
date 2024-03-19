"""
Construct TDA moments with periodic boundary conditions.
"""

import numpy as np
import scipy.special
from pyscf import lib
from pyscf.agf2 import mpi_helper
from pyscf.pbc import dft
from pyscf.pbc.gw.krgw_ac import get_qij

from momentGW import logging, mpi_helper, util
from momentGW.tda import dTDA as MoldTDA


class dTDA(MoldTDA):
    """
    Compute the self-energy moments using dTDA with periodic boundary
    conditions.

    Parameters
    ----------
    gw : BaseKGW
        GW object.
    nmom_max : int
        Maximum moment number to calculate.
    integrals : KIntegrals
        Density-fitted integrals at each k-point.
    mo_energy : dict, optional
        Molecular orbital energies at each k-point. Keys are "g" and
        "w" for the Green's function and screened Coulomb interaction,
        respectively. If `None`, use `gw.mo_energy` for both. Default
        value is `None`.
    mo_occ : dict, optional
        Molecular orbital occupancies at each k-point. Keys are "g"
        and "w" for the Green's function and screened Coulomb
        interaction, respectively. If `None`, use `gw.mo_occ` for both.
        Default value is `None`.
    """
    def __init__(
        self,
        gw,
        nmom_max,
        integrals,
        mo_energy=None,
        mo_occ=None,
        head_wings=False,
    ):
        super().__init__(gw, nmom_max, integrals, mo_energy, mo_occ)
        self.head_wings = head_wings
        if self.head_wings:
            q = np.array([1e-3, 0, 0]).reshape(1, 3)
            self.q_abs = self.kpts.cell.get_abs_kpts(q)
            self.qij = self.build_pert_term(self.q_abs[0])


    @logging.with_timer("Density-density moments")
    @logging.with_status("Constructing density-density moments")
    def build_dd_moments(self):
        """Build the moments of the density-density response.

        Returns
        -------
        moments : numpy.ndarray
            Moments of the density-density response at each k-point.
        """

        # Initialise the moments
        kpts = self.kpts
        moments = np.zeros((self.nkpts, self.nkpts, self.nmom_max + 1), dtype=object)


        if self.head_wings:
            head = np.zeros((self.nkpts, self.nmom_max + 1), dtype=object)
            M = np.zeros((self.nkpts), dtype=object)
            norm_q_abs = np.linalg.norm(self.q_abs[0])
            for kj in kpts.loop(1, mpi=True):
                kb = kpts.member(kpts.wrap_around(kpts[0] + kpts[kj]))
                M[kb] = self.integrals.Lia[kb, kb]

        # Get the zeroth order moment
        for q in kpts.loop(1):
            for kj in kpts.loop(1, mpi=True):
                kb = kpts.member(kpts.wrap_around(kpts[q] + kpts[kj]))
                moments[q, kb, 0] += self.integrals.Lia[kj, kb] / self.nkpts

            if self.head_wings:
                head[q, 0] += (np.sqrt(4. * np.pi) / norm_q_abs) * self.qij[q].conj()

        # Get the higher order moments
        for i in range(1, self.nmom_max + 1):
            for q in kpts.loop(1):
                for kj in kpts.loop(1, mpi=True):
                    kb = kpts.member(kpts.wrap_around(kpts[q] + kpts[kj]))

                    d = util.build_1h1p_energies(
                        (self.mo_energy_w[kj], self.mo_energy_w[kb]),
                        (self.mo_occ_w[kj], self.mo_occ_w[kb]),
                    )
                    moments[q, kb, i] += moments[q, kb, i - 1] * d.ravel()[None]
                    if self.head_wings and q==0:
                        head[kb, i] += head[kb, i - 1] * d.ravel()

                tmp = np.zeros((self.naux[q], self.naux[q]), dtype=complex)
                tmp_head = np.zeros((self.naux[q]), dtype=complex)

                for ki in kpts.loop(1, mpi=True):
                    ka = kpts.member(kpts.wrap_around(kpts[q] + kpts[ki]))

                    tmp += np.dot(moments[q, ka, i - 1], self.integrals.Lia[ki, ka].T.conj())

                    if q == 0 and self.head_wings:
                        tmp_head += lib.einsum("a,aP->P",head[kb, i - 1],  M[kb].T.conj())


                tmp = mpi_helper.allreduce(tmp)
                tmp *= 2.0
                tmp /= self.nkpts

                tmp_head = mpi_helper.allreduce(tmp_head)
                tmp_head *= 2.0
                tmp_head /= self.nkpts

                for kj in kpts.loop(1, mpi=True):
                    kb = kpts.member(kpts.wrap_around(kpts[q] + kpts[kj]))

                    moments[q, kb, i] += np.dot(tmp, self.integrals.Lai[kj, kb].conj())

                    if q == 0 and self.head_wings:
                        head[kb, i] += lib.einsum("P,Pa->a",tmp_head, M[kb])


        if self.head_wings:
            return {"moments":moments, "head":head}
        else:
            return moments

    def kernel(self, exact=False):
        """
        Run the polarizability calculation to compute moments of the
        self-energy.

        Parameters
        ----------
        exact : bool, optional
            Has no effect and is only present for compatibility with
            `dRPA`. Default value is `False`.

        Returns
        -------
        moments_occ : numpy.ndarray
            Moments of the occupied self-energy at each k-point.
        moments_vir : numpy.ndarray
            Moments of the virtual self-energy at each k-point.
        """
        return super().kernel(exact=exact)

    @logging.with_timer("Moment convolution")
    @logging.with_status("Convoluting moments")
    def convolve(self, eta, mo_energy_g=None, mo_occ_g=None):
        """
        Handle the convolution of the moments of the Green's function
        and screened Coulomb interaction.

        Parameters
        ----------
        eta : numpy.ndarray
            Moments of the density-density response partly transformed
            into moments of the screened Coulomb interaction at each
            k-point.
        mo_energy_g : numpy.ndarray, optional
            Energies of the Green's function at each k-point. If
            `None`, use `self.mo_energy_g`. Default value is `None`.
        mo_occ_g : numpy.ndarray, optional
            Occupancies of the Green's function at each k-point. If
            `None`, use `self.mo_occ_g`. Default value is `None`.

        Returns
        -------
        moments_occ : numpy.ndarray
            Moments of the occupied self-energy at each k-point.
        moments_vir : numpy.ndarray
            Moments of the virtual self-energy at each k-point.
        """

        # Get the orbitals
        if mo_energy_g is None:
            mo_energy_g = self.mo_energy_g
        if mo_occ_g is None:
            mo_occ_g = self.mo_occ_g
        kpts = self.kpts

        # Setup dependent on diagonal SE
        if self.gw.diagonal_se:
            pqchar = "p"
            fproc = lambda x: np.diag(x)
        else:
            pqchar = "pq"
            fproc = lambda x: x

        # We avoid self.nmo for inheritence reasons, but in MPI eta is
        # sparse, hence this weird code
        for part in eta.ravel():
            if isinstance(part, np.ndarray):
                nmo = part.shape[-1]
                break

        # Initialise the moments
        moments_occ = np.zeros((self.nkpts, self.nmom_max + 1, nmo, nmo), dtype=complex)
        moments_vir = np.zeros((self.nkpts, self.nmom_max + 1, nmo, nmo), dtype=complex)

        moms = np.arange(self.nmom_max + 1)
        for n in moms:
            # Get the binomial coefficients
            fp = scipy.special.binom(n, moms)
            fh = fp * (-1) ** moms

            for q in kpts.loop(1):
                for kp in kpts.loop(1, mpi=True):
                    kx = kpts.member(kpts.wrap_around(kpts[kp] - kpts[q]))
                    subscript = f"t,kt,kt{pqchar}->{pqchar}"

                    # Construct the occupied moments for this order
                    eo = np.power.outer(mo_energy_g[kx][mo_occ_g[kx] > 0], n - moms)
                    to = util.einsum(subscript, fh, eo, eta[kp, q][mo_occ_g[kx] > 0])
                    moments_occ[kp, n] += fproc(to)

                    # Construct the virtual moments for this order
                    ev = np.power.outer(mo_energy_g[kx][mo_occ_g[kx] == 0], n - moms)
                    tv = util.einsum(subscript, fp, ev, eta[kp, q][mo_occ_g[kx] == 0])
                    moments_vir[kp, n] += fproc(tv)

        # Numerical integration can lead to small non-hermiticity
        for n in range(self.nmom_max + 1):
            for k in kpts.loop(1, mpi=True):
                moments_occ[k, n] = 0.5 * (moments_occ[k, n] + moments_occ[k, n].T.conj())
                moments_vir[k, n] = 0.5 * (moments_vir[k, n] + moments_vir[k, n].T.conj())

        # Sum over all processes
        moments_occ = mpi_helper.allreduce(moments_occ)
        moments_vir = mpi_helper.allreduce(moments_vir)

        return moments_occ, moments_vir

    @logging.with_timer("Self-energy moments")
    @logging.with_status("Constructing self-energy moments")
    def build_se_moments(self, moments_dd):
        """Build the moments of the self-energy via convolution.

        Parameters
        ----------
        moments_dd : numpy.ndarray
            Moments of the density-density response at each k-point.

        Returns
        -------
        moments_occ : numpy.ndarray
            Moments of the occupied self-energy at each k-point.
        moments_vir : numpy.ndarray
            Moments of the virtual self-energy at each k-point.
        """

        kpts = self.kpts

        # Setup dependent on diagonal SE
        if self.gw.diagonal_se:
            pqchar = pchar = qchar = "p"
            eta_shape = lambda k: (self.mo_energy_g[k].size, self.nmom_max + 1, self.nmo)
        else:
            pqchar, pchar, qchar = "pq", "p", "q"
            eta_shape = lambda k: (self.mo_energy_g[k].size, self.nmom_max + 1, self.nmo, self.nmo)
        eta = np.zeros((self.nkpts, self.nkpts), dtype=object)

        if self.head_wings:
            cell = self.kpts.cell
            cell_vol = cell.vol
            total_vol = cell_vol * self.nkpts
            q0 = (6*np.pi/total_vol)**(1/3)
            norm_q_abs = np.linalg.norm(self.q_abs[0])
            eta_head = np.zeros((self.nkpts, self.nmom_max + 1), dtype=object)
            eta_wings = np.zeros((self.nkpts, self.nmom_max + 1), dtype=object)

        # Get the moments in (aux|aux) and rotate to (mo|mo)
        for n in range(self.nmom_max + 1):
            for q in kpts.loop(1):
                eta_aux = 0
                for kj in kpts.loop(1, mpi=True):
                    kb = kpts.member(kpts.wrap_around(kpts[q] + kpts[kj]))
                    if self.head_wings:
                        eta_aux += np.dot(moments_dd["moments"][q, kb, n], self.integrals.Lia[kj, kb].T.conj())
                        if q==0:
                            eta_head[kb, n] += -(np.sqrt(4. * np.pi) / norm_q_abs) * np.sum(moments_dd["head"][kb, n]*
                                                                                           self.qij[kb])
                            eta_wings[kb,n] += (np.sqrt(4. * np.pi) / norm_q_abs) * lib.einsum("Pa,a->P", moments_dd["moments"][q, kb, n],  self.qij[kb])
                    else:
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
                        eta[kp, q][x, n] += util.einsum(subscript, Lp, Lp.conj(), eta_aux)
                        if self.head_wings and q == 0:
                            original = eta[kp, q][x, n]

                            eta[kp, q][x, n] += (2/np.pi)*(q0)*eta_head[kp, n]*original

                            wing_tmp = lib.einsum("Pp,P->p", Lp, eta_wings[kp, n])
                            wing_tmp = wing_tmp.real*2
                            wing_tmp *= -(np.sqrt(cell_vol/ (4*(np.pi** 3))) * q0**2)

                            eta[kp, q][x, n] += lib.einsum("p,pq->pq", wing_tmp, original)

        cput1 = lib.logger.timer(self.gw, "rotating DD moments", *cput0)


        # Construct the self-energy moments
        moments_occ, moments_vir = self.convolve(eta)



        # if self.gw.fc:
        #     moments_dd_fc = self.build_dd_moments_fc()
        #
        #     moments_occ_fc, moments_vir_fc = self.build_se_moments_fc(*moments_dd_fc)
        #
        #     moments_occ += moments_occ_fc
        #     moments_vir += moments_vir_fc
        #
        #     cput1 = lib.logger.timer(self.gw, "fc correction", *cput1)

        return moments_occ, moments_vir

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
                    eta[:, t] += lib.einsum("xpL,L->xp",  integrals.Lpx[k, k].conj().T,moments_wing[k, t].conj())

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
            qij[k] = dens.flatten() / np.sqrt(self.gw.cell.vol)

        return qij



    build_dd_moments_exact = build_dd_moments

    @property
    def naux(self):
        """Number of auxiliaries."""
        return self.integrals.naux

    @property
    def nov(self):
        """Get the number of ov states in W."""
        return np.multiply.outer(
            [np.sum(occ > 0) for occ in self.mo_occ_w],
            [np.sum(occ == 0) for occ in self.mo_occ_w],
        )

    @property
    def kpts(self):
        """Get the k-points."""
        return self.gw.kpts

    @property
    def nkpts(self):
        """Get the number of k-points."""
        return self.gw.nkpts
