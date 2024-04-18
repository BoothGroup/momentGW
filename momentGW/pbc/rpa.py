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
                    np.sum(np.abs(self.integrals.Lia[ki, kb]) ** 2, axis=0) / self.nkpts
                )

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
        quad = self.optimise_offset_quad(d, diag_eri)

        # Perform the offset integral
        offset = self.eval_offset_integral(quad, d)

        # Get the main integral quadrature
        quad = self.optimise_main_quad(d, diag_eri)

        # Perform the main integral
        integral = self.eval_main_integral(quad, d)

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

        return integral[0] + offset

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
        naux = self.naux
        Lia = self.integrals.Lia
        moments = np.zeros((self.nkpts, self.nkpts, self.nmom_max + 1), dtype=object)

        # Construct the energy differences
        d = self._build_d()

        # Calculate (L|ia) D_{ia} and (L|ia) D_{ia}^{-1} intermediates
        Liad = self._build_Liad(Lia, d)
        Liadinv = self._build_Liadinv(Lia, d)

        for q in kpts.loop(1):
            # Get the zeroth order moment
            tmp = np.zeros((naux[q], naux[q]), dtype=complex)
            inter = 0.0
            for kj in kpts.loop(1, mpi=True):
                kb = kpts.member(kpts.wrap_around(kpts[q] + kpts[kj]))
                tmp += np.dot(Liadinv[q, kb], self.integrals.Lia[kj, kb].conj().T)
                inter += np.dot(integral[q, kb], Liadinv[q, kb].T.conj())
            tmp = mpi_helper.allreduce(tmp)
            inter = mpi_helper.allreduce(inter)
            tmp *= 2.0
            u = np.linalg.inv(np.eye(tmp.shape[0]) * self.nkpts / 2 + tmp)

            rest = np.dot(inter, u) * self.nkpts / 2
            for ki in kpts.loop(1, mpi=True):
                ka = kpts.member(kpts.wrap_around(kpts[q] + kpts[ki]))
                moments[q, ka, 0] = integral[q, ka] / d[q, ka] * self.nkpts / 2
                moments[q, ka, 0] -= np.dot(rest, Liadinv[q, ka]) * 2

        # Get the first order moment
        moments[:, :, 1] = Liad / self.nkpts

        # Get the higher order moments
        for i in range(2, self.nmom_max + 1):
            for q in kpts.loop(1):
                tmp = 0.0
                for ka in kpts.loop(1, mpi=True):
                    kb = kpts.member(kpts.wrap_around(kpts[q] + kpts[ka]))
                    moments[q, kb, i] = moments[q, kb, i - 2] * d[q, kb] ** 2
                    tmp += np.dot(moments[q, kb, i - 2], self.integrals.Lia[ka, kb].conj().T)
                tmp = mpi_helper.allreduce(tmp)
                tmp /= self.nkpts
                tmp *= 2
                for ki in kpts.loop(1, mpi=True):
                    ka = kpts.member(kpts.wrap_around(kpts[q] + kpts[ki]))
                    moments[q, ka, i] += np.dot(tmp, Liad[q, ka]) * 2

        return moments

    def optimise_offset_quad(self, d, diag_eri, name="main"):
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
            Name of the integral. Default value is `"main"`.

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
        naux = self.naux
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

                tmp = np.linalg.inv(np.eye(naux[q]) + qz) - np.eye(naux[q])
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
