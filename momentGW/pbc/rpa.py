"""
Construct RPA moments with periodic boundary conditions.
"""

import numpy as np

from momentGW import logging, mpi_helper, util
from momentGW.pbc.tda import dTDA
from momentGW.rpa import dRPA as MoldRPA


class dRPA(dTDA, MoldRPA):
    """
    Compute the self-energy moments using dRPA and numerical integration
    with periodic boundary conditions.

    Parameters
    ----------
    gw : BaseKGW
        GW object.
    nmom_max : int
        Maximum moment number to calculate.
    integrals : KIntegrals
        Density-fitted integrals at each k-point.
    mo_energy : dict, optional
        Molecular orbital energies. Keys are "g" and "w" for the Green's
        function and screened Coulomb interaction, respectively.
        If `None`, use `gw.mo_energy` for both. Default value is `None`.
    mo_occ : dict, optional
        Molecular orbital occupancies. Keys are "g" and "w" for the
        Green's function and screened Coulomb interaction, respectively.
        If `None`, use `gw.mo_occ` for both. Default value is `None`.

    Notes
    -----
    See `momentGW.tda.dTDA.__init__` for initialisation details and
    `momentGW.tda.dTDA.kernel` for calculation run details.
    """
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
            Moments of the occupied self-energy.
        moments_vir : numpy.ndarray
            Moments of the virtual self-energy.
        """
        if self.e_corr:
            d = self._build_d()
            diag_eri = self._build_diag_eri()
            quad_main = self.optimise_main_quad(d, diag_eri)
            e_corr = self.correlation_energy(quad_main, d)

            print("Correlation energy: ", e_corr)
            return None, None


        # Build the density-density response moments
        moments_dd = self.build_dd_moments()

        # Build the self-energy moments
        moments_occ, moments_vir = self.build_se_moments(moments_dd)

        return moments_occ, moments_vir

    def _build_d(self):
        """Construct the energy differences matrix.

        Returns
        -------
        d : numpy.ndarray
            Orbital energy differences at each k-point.
        """

        d = np.zeros((self.nkpts, self.nkpts), dtype=object)

        for q in self.kpts.loop(1):
            for ki in self.kpts.loop(1, mpi=True):
                ka = self.kpts.member(self.kpts.wrap_around(self.kpts[q] + self.kpts[ki]))
                d[q, ka] = util.build_1h1p_energies(
                    (self.mo_energy_w[ki], self.mo_energy_w[ka]),
                    (self.mo_occ_w[ki], self.mo_occ_w[ka]),
                ).ravel()

        return d

    def _build_diag_eri(self):
        """Construct the diagonal of the ERIs at each k-point.

        Returns
        -------
        diag_eri : numpy.ndarray
            Diagonal of the ERIs at each k-point.
        """

        diag_eri = np.zeros((self.nkpts, self.nkpts), dtype=object)

        for q in self.kpts.loop(1):
            for ki in self.kpts.loop(1, mpi=True):
                kb = self.kpts.member(self.kpts.wrap_around(self.kpts[q] + self.kpts[ki]))
                diag_eri[q, kb] = (
                        util.einsum("np,np->p",
                                    np.abs(self.integrals.Lia[ki, kb]),
                                    np.abs(self.integrals.Lia[
                                               ki, kb])) / self.nkpts
                )
                # if self.fc and q==0:
                #     diag_eri[q, kb] += (
                #     util.einsum("p,p->p",
                #                 np.abs(self.qij[ki].conj()),
                #                 np.abs(self.qij[ki]))
                #     ) * 4.0 * np.pi / np.linalg.norm(self.q_abs[0])**2
                #     diag_eri[q, kb] += (
                #                            util.einsum("p,np->p",
                #                                        np.abs(self.qij[
                #                                                   ki].conj()),
                #                                        np.abs(
                #                                            self.integrals.Lia[
                #                                                ki, kb]))
                #                        ) * np.sqrt(4.0 * np.pi) / np.linalg.norm(
                #         self.q_abs[0])

        return diag_eri

    def _build_Liad(self, Lia, d):
        """Construct the ``Liad`` array.

        Returns
        -------
        Liad : numpy.ndarray
           Product of Lia and the orbital energy differences at each
           k-point.
        """

        Liad = np.zeros((self.nkpts, self.nkpts), dtype=object)

        for q in self.kpts.loop(1):
            for ki in self.kpts.loop(1, mpi=True):
                kb = self.kpts.member(self.kpts.wrap_around(self.kpts[q] + self.kpts[ki]))
                Liad[q, kb] = Lia[ki, kb] * d[q, kb]

        return Liad

    def _build_Liadinv(self, Lia, d):
        """Construct the ``Liadinv`` array.

        Returns
        -------
        Liadinv : numpy.ndarray
           Division of Lia and the orbital energy differences at each
           k-point.
        """

        Liadinv = np.zeros((self.nkpts, self.nkpts), dtype=object)

        for q in self.kpts.loop(1):
            for ki in self.kpts.loop(1, mpi=True):
                kb = self.kpts.member(self.kpts.wrap_around(self.kpts[q] + self.kpts[ki]))
                Liadinv[q, kb] = Lia[ki, kb] / d[q, kb]

        return Liadinv

    def _build_diag_head_eri(self):
        """Construct the diagonal of the Head ERIs at each k-point.

        Returns
        -------
        diag_head_eri : numpy.ndarray
            Diagonal of the Head ERIs at each k-point.
        """

        diag_head_eri = np.zeros((self.nkpts), dtype=object)

        for ki in self.kpts.loop(1, mpi=True):
            diag_head_eri[ki] = (
                    util.einsum("p,p->p",
                                np.abs(self.qij[ki].conj()),
                                np.abs(self.qij[ki]))
            )
        diag_head_eri *= 4.0 * np.pi / np.linalg.norm(self.q_abs[0])**2
        return diag_head_eri

    def _build_qijd(self, d):
        """Construct the ``qijd`` array.

        Returns
        -------
        qijd : numpy.ndarray
           Product of qij and the orbital energy differences at each
           k-point.
        """

        qijd = np.zeros((self.nkpts), dtype=object)

        for ki in self.kpts.loop(1, mpi=True):
            qijd[ki] = self.qij[ki] * d[0, ki]

        qijd *= np.sqrt(4.0 * np.pi) / np.linalg.norm(self.q_abs[0])
        return qijd

    @logging.with_timer("Numerical integration")
    @logging.with_status("Performing numerical integration")
    def integrate(self):
        """
        Optimise the quadrature and perform the integration for a given
        set of k-points for the zeroth moment.

        Returns
        -------
        integral : numpy.ndarray
            Integral array, including the offset part.
        """

        # Construct the energy differences
        d = self._build_d()

        # Calculate diagonal part of ERIs
        diag_eri = self._build_diag_eri()

        # Get the offset integral quadrature
        quad_offset = self.optimise_offset_quad(d, diag_eri)

        # Perform the offset integral
        offset = self.eval_offset_integral(quad_offset, d)

        # Get the main integral quadrature
        quad_main = self.optimise_main_quad(d, diag_eri)

        # Perform the main integral
        integral = self.eval_main_integral(quad_main, d)

        # Report quadrature error
        if self.report_quadrature_error:
            a = 0.0
            b = 0.0
            for q in self.kpts.loop(1):
                for ka in self.kpts.loop(1, mpi=True):
                    a += np.sum((integral[0, q, ka] - integral[2, q, ka]) ** 2)
                    b += np.sum((integral[0, q, ka] - integral[1, q, ka]) ** 2)
            a, b = mpi_helper.allreduce(np.array([a, b]))
            a, b = a**0.5, b**0.5
            err = self.estimate_error_clencur(a, b)
            style_half = logging.rate(a, 1e-4, 1e-3)
            style_quar = logging.rate(b, 1e-8, 1e-6)
            style_full = logging.rate(err, 1e-12, 1e-9)
            logging.write(
                f"Error in integral:  [{style_full}]{err:.3e}[/] "
                f"(half = [{style_half}]{a:.3e}[/], quarter = [{style_quar}]{b:.3e}[/])",
            )

        if self.fc:
            # Calculate diagonal part of Head ERIs
            # diag_head_eri = self._build_diag_head_eri()
            #
            # # Get the Head offset integral quadrature
            # quad_head = self.optimise_head_offset_quad(d, diag_head_eri)

            # Perform the Head offset integral
            offset_head = self.eval_head_offset_integral(quad_offset, d)

            # Get the Head main integral quadrature
            # quad_head = self.optimise_head_main_quad(d, diag_eri)

            # Perform the Head main integral
            integral_head = self.eval_head_main_integral(quad_main, d)

        if self.fc:
            return {"moments": integral[0] + offset, "head": integral_head[0]+offset_head}
        else:
            return {"moments":integral[0] + offset}

    @logging.with_timer("Density-density moments")
    @logging.with_status("Constructing density-density moments")
    def build_dd_moments(self, integral=None):
        """Build the moments of the density-density response.

        Parameters
        ----------
        integral : numpy.ndarray, optional
            Integral array, including the offset part. If `None`,
            calculate from scratch. Default is `None`.

        Returns
        -------
        moments : numpy.ndarray
            Moments of the density-density response.
        """

        if integral is None:
            integral = self.integrate()

        kpts = self.kpts
        Lia = self.integrals.Lia
        moments = np.zeros((self.nkpts, self.nkpts, self.nmom_max + 1), dtype=object)
        if self.fc:
            head = np.zeros((self.nkpts, self.nmom_max + 1), dtype=object)

        # Construct the energy differences
        d = self._build_d()

        # Calculate (L|ia) D_{ia} and (L|ia) D_{ia}^{-1} intermediates
        Liad = self._build_Liad(Lia, d)
        Liadinv = self._build_Liadinv(Lia, d)

        for q in kpts.loop(1):
            # Get the zeroth order moment
            tmp = np.zeros((self.naux[q], self.naux[q]), dtype=complex)
            inter = 0.0
            inter_head = 0.0
            for kj in kpts.loop(1, mpi=True):
                kb = kpts.member(kpts.wrap_around(kpts[q] + kpts[kj]))
                tmp += np.dot(Liadinv[q, kb], self.integrals.Lia[kj, kb].conj().T)
                inter += np.dot(integral["moments"][q, kb], Liadinv[q, kb].T.conj())
                if self.fc and q == 0:
                    inter_head += util.einsum("i,ia->a", integral["head"][kj],
                                              Liadinv[0, kj].T.conj())

            tmp = mpi_helper.allreduce(tmp)
            inter = mpi_helper.allreduce(inter)
            inter_head = mpi_helper.allreduce(inter_head)
            tmp *= 2.0
            u = np.linalg.inv(np.eye(tmp.shape[0]) * self.nkpts / 2 + tmp)

            rest = np.dot(inter, u) * self.nkpts / 2
            if self.fc and q == 0:
                rest_head = np.dot(inter_head, u) * self.nkpts / 2
            for ki in kpts.loop(1, mpi=True):
                ka = kpts.member(kpts.wrap_around(kpts[q] + kpts[ki]))
                moments[q, ka, 0] = integral["moments"][q, ka] / d[q, ka] * self.nkpts / 2
                moments[q, ka, 0] -= np.dot(rest, Liadinv[q, ka]) * 2
                if self.fc and q==0:
                    head[ka, 0] = integral["head"][ki] / d[0, ki] * self.nkpts / 2
                    head[ka, 0] -= np.dot(rest_head, Liadinv[0, ki]) * 2

        # Get the first order moment
        moments[:, :, 1] = Liad / self.nkpts
        if self.fc:
            head[:,1] = self._build_qijd(d)

        # Get the higher order moments
        for i in range(2, self.nmom_max + 1):
            for q in kpts.loop(1):
                tmp = 0.0
                tmp_head = 0.0
                for ka in kpts.loop(1, mpi=True):
                    kb = kpts.member(kpts.wrap_around(kpts[q] + kpts[ka]))
                    moments[q, kb, i] = moments[q, kb, i - 2] * d[q, kb] ** 2
                    tmp += np.dot(moments[q, kb, i - 2], self.integrals.Lia[ka, kb].conj().T)
                    if q == 0 and self.fc:
                        head[ka, i] = head[ka, i - 2] * d[0, ka] ** 2
                        tmp_head += util.einsum(
                            "a,aP->P", head[ka, i - 1],
                            self.integrals.Lia[ka, ka].T.conj()
                        )
                tmp = mpi_helper.allreduce(tmp)
                tmp /= self.nkpts
                tmp *= 2

                tmp_head = mpi_helper.allreduce(tmp_head)
                tmp_head *= 2.0
                tmp_head /= self.nkpts
                for ki in kpts.loop(1, mpi=True):
                    ka = kpts.member(kpts.wrap_around(kpts[q] + kpts[ki]))
                    moments[q, ka, i] += np.dot(tmp, Liad[q, ka]) * 2
                    if q == 0 and self.fc:
                        head[ka, i] += util.einsum("P,Pa->a", tmp_head, Liad[0, ka]) * 2

        if self.fc:
            return {"moments": moments, "head": head}
        else:
            return moments

    def optimise_offset_quad(self, d, diag_eri, name="offset"):
        """
        Optimise the grid spacing of Gauss-Laguerre quadrature for the
        offset integral.

        Parameters
        ----------
        d : numpy.ndarray
            Orbital energy differences at each k-point.
        diag_eri : numpy.ndarray
            Diagonal of the ERIs at each k-point.
        name : str, optional
            Name of the integral. Default value is `"offset"`.

        Returns
        -------
        points : numpy.ndarray
            The quadrature points.
        weights : numpy.ndarray
            The quadrature weights.
        """

        # Generate the bare quadrature
        bare_quad = self.gen_gausslag_quad_semiinf()

        # Calculate the exact value of the integral for the diagonal
        exact = 0.0
        for q in self.kpts.loop(1):
            for ki in self.kpts.loop(1, mpi=True):
                ka = self.kpts.member(self.kpts.wrap_around(self.kpts[q] + self.kpts[ki]))
                exact += np.dot(1.0 / d[q, ka], d[q, ka] * diag_eri[q, ka])
        exact = mpi_helper.allreduce(exact)

        # Define the integrand
        integrand = lambda quad: self.eval_diag_offset_integral(quad, d, diag_eri)

        # Get the optimal quadrature
        quad = self.get_optimal_quad(bare_quad, integrand, exact, name=name)

        return quad

    def eval_diag_offset_integral(self, quad, d, diag_eri):
        """Evaluate the diagonal of the offset integral.

        Parameters
        ----------
        quad : tuple
            The quadrature points and weights.
        d : numpy.ndarray
            Orbital energy differences at each k-point.
        diag_eri : numpy.ndarray
            Diagonal of the ERIs at each k-point.

        Returns
        -------
        integral : numpy.ndarray
            Offset integral.
        """

        # Calculate the integral for each point
        integral = 0.0
        for point, weight in zip(*quad):
            for q in self.kpts.loop(1):
                for ki in self.kpts.loop(1, mpi=True):
                    ka = self.kpts.member(self.kpts.wrap_around(self.kpts[q] + self.kpts[ki]))
                    tmp = d[q, ka] * diag_eri[q, ka]
                    expval = np.exp(-2 * point * d[q, ka])
                    res = np.dot(expval, tmp)
                    integral += 2 * res * weight
        integral = mpi_helper.allreduce(integral)

        return integral

    def eval_offset_integral(self, quad, d, Lia=None):
        """Evaluate the offset integral.

        Parameters
        ----------
        quad : tuple
            The quadrature points and weights.
        d : numpy.ndarray
            Orbital energy differences at each k-point.
        Lia : dict of numpy.ndarray
            Dict. with keys that are pairs of k-point indices (Nkpt, Nkpt)
            with an array of form (aux, W occ, W vir) at this k-point pair.
            The 1st Nkpt is defined by the difference between k-points and
            the second index's kpoint. If `None`, use `self.integrals.Lia`.
        Liad : dict of numpy.ndarray
            Product of Lia and the orbital energy differences at each
            k-point.


        Returns
        -------
        integral : numpy.ndarray
            Offset integral.
        """

        # Get the integral intermediates
        if Lia is None:
            Lia = self.integrals.Lia
        Liad = self._build_Liad(Lia, d)
        integrals = 2 * Liad / (self.nkpts**2)

        kpts = self.kpts

        # Calculate the integral for each point
        for point, weight in zip(*quad):
            for q in kpts.loop(1):
                lhs = 0.0
                for ka in kpts.loop(1, mpi=True):
                    kb = kpts.member(kpts.wrap_around(kpts[q] + kpts[ka]))
                    expval = np.exp(-point * d[q, kb])
                    lhs += np.dot(Liad[q, kb] * expval[None], Lia[ka, kb].T.conj())
                lhs = mpi_helper.allreduce(lhs)
                lhs /= self.nkpts
                lhs *= 2

                for ka in kpts.loop(1, mpi=True):
                    kb = kpts.member(kpts.wrap_around(kpts[q] + kpts[ka]))
                    rhs = self.integrals.Lia[ka, kb] * np.exp(-point * d[q, kb])
                    rhs /= self.nkpts**2
                    res = np.dot(lhs, rhs)
                    integrals[q, kb] += res * weight * 4

        return integrals

    def optimise_main_quad(self, d, diag_eri, name="main"):
        """
        Optimise the grid spacing of Clenshaw-Curtis quadrature for the
        main integral.

        Parameters
        ----------
        d : numpy.ndarray
            Orbital energy differences at each k-point.
        diag_eri : numpy.ndarray
            Diagonal of the ERIs at each k-point.

        Returns
        -------
        points : numpy.ndarray
            The quadrature points.
        weights : numpy.ndarray
            The quadrature weights.
        """

        # Generate the bare quadrature
        bare_quad = self.gen_clencur_quad_inf(even=True)

        # Calculate the exact value of the integral for the diagonal
        exact = 0.0
        d_sq = np.zeros((self.nkpts, self.nkpts), dtype=object)
        d_eri = np.zeros((self.nkpts, self.nkpts), dtype=object)
        d_sq_eri = np.zeros((self.nkpts, self.nkpts), dtype=object)
        for q in self.kpts.loop(1):
            for kj in self.kpts.loop(1, mpi=True):
                kb = self.kpts.member(self.kpts.wrap_around(self.kpts[q] + self.kpts[kj]))
                exact += np.sum((d[q, kb] * (d[q, kb] + diag_eri[q, kb])) ** 0.5)
                exact -= 0.5 * np.dot(1.0 / d[q, kb], d[q, kb] * diag_eri[q, kb])
                exact -= np.sum(d[q, kb])
                d_sq[q, kb] = d[q, kb] ** 2
                d_eri[q, kb] = d[q, kb] * diag_eri[q, kb]
                d_sq_eri[q, kb] = d[q, kb] * (d[q, kb] + diag_eri[q, kb])
        exact = mpi_helper.allreduce(exact)

        # Define the integrand
        integrand = lambda quad: self.eval_diag_main_integral(quad, d, d_sq, d_eri, d_sq_eri)

        # Get the optimal quadrature
        quad = self.get_optimal_quad(bare_quad, integrand, exact, name=name)

        return quad

    def eval_diag_main_integral(self, quad, d, d_sq, d_eri, d_sq_eri):
        """Evaluate the diagonal of the main integral.

        Parameters
        ----------
        quad : tuple
            The quadrature points and weights.
        d : numpy.ndarray
            Orbital energy differences at each k-point.
        d_sq : numpy.ndarray
            Orbital energy differences squared at each k-point.
            See "optimise_main_quad" for more details.
        d_eri : numpy.ndarray
            Orbital energy differences times the diagonal of the
            ERIs at each k-point.
            See "optimise_main_quad" for more details.
        d_sq_eri : numpy.ndarray
            Orbital energy differences times the diagonal of the ERIs plus
            the orbital energy differences at each k-point.
            See "optimise_main_quad" for more details.

        Returns
        -------
        integral : numpy.ndarray
            Main integral.
        """

        integral = 0.0

        def diag_contrib(x, freq):
            integral = x / (x + freq**2)
            integral /= np.pi
            return integral

        # Calculate the integral for each point
        for point, weight in zip(*quad):
            contrib = 0.0
            for q in self.kpts.loop(1):
                for kj in self.kpts.loop(1, mpi=True):
                    kb = self.kpts.member(self.kpts.wrap_around(self.kpts[q] + self.kpts[kj]))
                    contrib_int = diag_contrib(d_sq_eri[q, kb], point)
                    contrib_int -= diag_contrib(d_sq[q, kb], point)
                    contrib += np.abs(np.sum(contrib_int))

                    f_sq = 1.0 / (d[q, kb] ** 2 + point**2) ** 2
                    contrib -= np.abs(point**2 * np.dot(f_sq, d_eri[q, kb]) / np.pi)

            integral += weight * contrib

        integral = mpi_helper.allreduce(integral)

        return integral

    def eval_main_integral(self, quad, d, Lia=None):
        """Evaluate the main integral.

        Parameters
        ----------
        quad : tuple
            The quadrature points and weights.

        Variables
        ----------
        d : numpy.ndarray
            Orbital energy differences at each k-point.
        Lia : dict of numpy.ndarray
            Dict. with keys that are pairs of k-point indices (Nkpt, Nkpt)
            with an array of form (aux, W occ, W vir) at this k-point pair.
            The 1st Nkpt is defined by the difference between k-points and
            the second index's kpoint. If `None`, use `self.integrals.Lia`.
        Liad : dict of numpy.ndarray
            Product of Lia and the orbital energy differences at each
            k-point.

        Returns
        -------
        integral : numpy.ndarray
            Offset integral.
        """

        # Get the integral intermediates
        if Lia is None:
            Lia = self.integrals.Lia
        Liad = self._build_Liad(Lia, d)

        # Initialise the integral
        dim = 3 if self.report_quadrature_error else 1
        integral = np.zeros((dim, self.nkpts, self.nkpts), dtype=object)

        # Calculate the integral for each point
        kpts = self.kpts
        for i, (point, weight) in enumerate(zip(*quad)):
            contrib = np.zeros_like(d, dtype=object)

            for q in kpts.loop(1):
                f = np.zeros((self.nkpts), dtype=object)
                qz = 0.0
                for ki in kpts.loop(1, mpi=True):
                    kj = kpts.member(kpts.wrap_around(kpts[q] + kpts[ki]))
                    f[kj] = 1.0 / (d[q, kj] ** 2 + point**2)
                    pre = (Lia[ki, kj] * f[kj]) * (4 / self.nkpts)
                    qz += np.dot(pre, Liad[q, kj].T.conj())
                qz = mpi_helper.allreduce(qz)

                tmp = np.linalg.inv(np.eye(self.naux[q]) + qz) - np.eye(self.naux[q])
                inner = np.dot(qz, tmp)

                for ka in kpts.loop(1, mpi=True):
                    kb = kpts.member(kpts.wrap_around(kpts[q] + kpts[ka]))
                    contrib[q, kb] = 2 * np.dot(inner, Lia[ka, kb]) / (self.nkpts**2)
                    value = weight * (contrib[q, kb] * f[kb] * (point**2 / np.pi))

                    integral[0, q, kb] += value
                    if i % 2 == 0 and self.report_quadrature_error:
                        integral[1, q, kb] += 2 * value
                    if i % 4 == 0 and self.report_quadrature_error:
                        integral[2, q, kb] += 4 * value

        return integral


    def eval_head_offset_integral(self, quad, d, Lia=None):
        """Evaluate the offset integral.

        Parameters
        ----------
        quad : tuple
            The quadrature points and weights.
        d : numpy.ndarray
            Orbital energy differences at each k-point.
        Lia : dict of numpy.ndarray
            Dict. with keys that are pairs of k-point indices (Nkpt, Nkpt)
            with an array of form (aux, W occ, W vir) at this k-point pair.
            The 1st Nkpt is defined by the difference between k-points and
            the second index's kpoint. If `None`, use `self.integrals.Lia`.
        Liad : dict of numpy.ndarray
            Product of Lia and the orbital energy differences at each
            k-point.


        Returns
        -------
        integral : numpy.ndarray
            Offset integral.
        """

        # Get the integral intermediates
        if Lia is None:
            Lia = self.integrals.Lia
        qijd = self._build_qijd(d)
        integrals = 2 * qijd / (self.nkpts) # may not need this nkpts

        kpts = self.kpts

        # Calculate the integral for each point
        for point, weight in zip(*quad):
            lhs = 0.0
            for ka in kpts.loop(1, mpi=True):
                expval = np.exp(-point * d[0, ka])
                lhs += util.einsum("a,aP->P",
                                   qijd[ka] * expval,
                                   Lia[ka, ka].T.conj())
            lhs = mpi_helper.allreduce(lhs)
            lhs /= self.nkpts
            lhs *= 2

            for ka in kpts.loop(1, mpi=True):
                rhs = self.integrals.Lia[ka, ka] * np.exp(-point * d[0, ka])
                rhs /= self.nkpts**2
                res = util.einsum("P,Pa->a",lhs, rhs)
                integrals[ka] += res * weight * 4

        return integrals


    def eval_head_main_integral(self, quad, d, Lia=None):
        """Evaluate the main integral.

        Parameters
        ----------
        quad : tuple
            The quadrature points and weights.

        Variables
        ----------
        d : numpy.ndarray
            Orbital energy differences at each k-point.
        Lia : dict of numpy.ndarray
            Dict. with keys that are pairs of k-point indices (Nkpt, Nkpt)
            with an array of form (aux, W occ, W vir) at this k-point pair.
            The 1st Nkpt is defined by the difference between k-points and
            the second index's kpoint. If `None`, use `self.integrals.Lia`.
        Liad : dict of numpy.ndarray
            Product of Lia and the orbital energy differences at each
            k-point.

        Returns
        -------
        integral : numpy.ndarray
            Offset integral.
        """

        # Get the integral intermediates
        if Lia is None:
            Lia = self.integrals.Lia
        Liad = self._build_Liad(Lia, d)

        # Initialise the integral
        dim = 3 if self.report_quadrature_error else 1
        integral = np.zeros((dim, self.nkpts), dtype=object)

        # Calculate the integral for each point
        kpts = self.kpts

        for i, (point, weight) in enumerate(zip(*quad)):
            contrib = np.zeros_like(self.qij, dtype=object)

            f = np.zeros((self.nkpts), dtype=object)
            qz = 0.0
            qz_head = 0.0
            for ki in kpts.loop(1, mpi=True):
                f[ki] = 1.0 / (d[0, ki] ** 2 + point**2)
                pre = (Lia[ki, ki] * f[ki]) * (4 / self.nkpts)
                pre_head = (self.qij[ki] * f[ki]) * (4)  # check constants
                pre_head *= np.sqrt(4.0 * np.pi) / np.linalg.norm(
                    self.q_abs[0])
                qz += np.dot(pre, Liad[0, ki].T.conj())
                qz_head += util.einsum("a,aP->P", pre_head, Liad[0, ki].T.conj())
            qz = mpi_helper.allreduce(qz)
            qz_head = mpi_helper.allreduce(qz_head)

            tmp = np.linalg.inv(np.eye(self.naux[0]) + qz) - np.eye(self.naux[0])
            inner = util.einsum("i,ia->a",qz_head, tmp)

            for ka in kpts.loop(1, mpi=True):
                contrib[ka] = (2 * util.einsum("i,ia->a", inner, Lia[ka, ka])
                               / (self.nkpts**2))
                value = weight * (contrib[ka] * f[ka] * (point**2 / np.pi))

                integral[0, ka] += value
                # if i % 2 == 0 and self.report_quadrature_error:
                #     integral[1, q, kb] += 2 * value
                # if i % 4 == 0 and self.report_quadrature_error:
                #     integral[2, q, kb] += 4 * value

        return integral

    def correlation_energy(self,quad, d, Lia=None):
        """Calculate the correlation energy."""
        if Lia is None:
            Lia = self.integrals.Lia
        Liad = self._build_Liad(Lia, d)

        # Initialise the integral
        E_corr = 0.0j

        # Calculate the integral for each point
        kpts = self.kpts
        for i, (point, weight) in enumerate(zip(*quad)):

            for q in kpts.loop(1):
                f = np.zeros((self.nkpts), dtype=object)
                qz = 0.0
                if self.fc and q==0:
                    qz_head = 0.0
                    qz_wings = 0.0
                for ki in kpts.loop(1, mpi=True):
                    kj = kpts.member(kpts.wrap_around(kpts[q] + kpts[ki]))
                    f[kj] = 1.0 / (d[q, kj] ** 2 + point ** 2)
                    pre = (Lia[ki, kj] * f[kj]) * (4 / self.nkpts)
                    qz += np.dot(pre, Liad[q, kj].T.conj())
                    if self.fc and q==0:
                        lhs = (self.qij[ki] * f[ki]) * (4/self.nkpts)
                        lhs *= np.sqrt(4.0 * np.pi) / np.linalg.norm(self.q_abs[0])
                        rhs = self.qij[ki] * d[0, ki]
                        rhs *= np.sqrt(4.0 * np.pi) / np.linalg.norm(self.q_abs[0])
                        qz_head = util.einsum("a,a->", lhs, rhs.T.conj())
                        qz_wings = util.einsum("Pa,a->P", pre, rhs.T.conj())

                tmp = np.linalg.inv(
                    np.eye(self.naux[q]) + qz) - np.eye(
                    self.naux[q])

                if q == 0 and self.fc:
                    eps_inv = np.zeros((self.naux[q] + 1,self.naux[q] + 1),dtype=tmp.dtype)

                    inv_Pi = np.linalg.inv(np.eye(self.naux[q]) + qz)

                    eps_inv_PQ = inv_Pi #- np.eye(self.naux[q])

                    temp = np.einsum("P,PQ,Q->", qz_wings.conj(),
                                     np.linalg.inv(np.eye(self.naux[q]) + qz),
                                     qz_wings)

                    eps_inv_00 = 1/((1+qz_head) - temp)
                    eps_inv_P0 = (-eps_inv_00) * np.dot(eps_inv_PQ, qz_wings)

                    eps_inv[0,0] = eps_inv_00
                    eps_inv[0, 1:] = eps_inv_P0.conj()
                    eps_inv[1:,0] = eps_inv_P0

                    pre_extra = np.dot(inv_Pi, qz_wings)
                    extra = util.einsum("P,Q->PQ", pre_extra*eps_inv_00,
                                pre_extra.T.conj())
                    eps_inv[1:,1:] = inv_Pi + extra

                cell_vol = self.kpts.cell.vol
                total_vol = cell_vol * self.nkpts

                q0 = (6 * np.pi ** 2 / total_vol) ** (1 / 3)

                for ka in kpts.loop(1, mpi=True):
                    kb = kpts.member(kpts.wrap_around(kpts[q] + kpts[ka]))
                    if q == 0 and self.fc:
                        inner = np.zeros((self.naux[q] + 1,self.naux[q] + 1),dtype=eps_inv.dtype)
                        rhs = util.einsum("j,jQ->jQ", 2 *(point ** 2) * f[kb],
                                          Liad[q, kb].T.conj())
                        lhs = util.einsum("Pi,i->Pi",
                                          Lia[ka, kb], 2 * f[kb])
                        inner_PQ = util.einsum("Pi,iQ->PQ", lhs,
                                            rhs) / self.nkpts ** 2

                        rhs_hw = util.einsum("j,j->j", 2 *(point ** 2) * f[kb],
                                                (self.qij[kb] * d[0, kb]).T.conj())
                        rhs_hw *= np.sqrt(4.0 * np.pi) / np.linalg.norm(self.q_abs[0])
                        lhs_hw = (self.qij[kb] * 2 * f[kb])
                        lhs_hw *= np.sqrt(4.0 * np.pi) / np.linalg.norm(self.q_abs[0])

                        inner_head = util.einsum("i,j->", lhs_hw, rhs_hw)
                        inner_wings = util.einsum("Pi,i->P", lhs, rhs_hw)

                        inner_head *= (2 / np.pi) * (q0)*(1/self.nkpts)
                        inner_wings *= (np.sqrt(cell_vol / (4 * (np.pi**3))) * q0**2)

                        inner[0,0] = inner_head
                        inner[0, 1:] = inner_wings.conj()
                        inner[1:, 0] = inner_wings
                        inner[1:,1:] = inner_PQ

                        E_corr += weight * np.trace(
                            util.einsum("PQ,QR->PR", (eps_inv-np.eye(self.naux[q]+1)),
                                        inner)) * (
                                    1 / np.pi)

                    else:
                        rhs = util.einsum("j,jQ->jQ", 2 * (point ** 2) * f[kb],
                                          Liad[q, kb].T.conj())
                        lhs = util.einsum("Pi,i->Pi",
                                          Lia[ka, kb], 2* f[kb], )
                        inner = util.einsum("Pi,iQ->PQ", lhs,
                                            rhs) / self.nkpts ** 2
                        value = util.einsum("PQ,QR->PR", tmp,inner)
                        E_corr += weight * np.trace(value)* (1 / np.pi)

        return 0.5*E_corr
