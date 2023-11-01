"""
Construct RPA moments with periodic boundary conditions.
"""

import numpy as np
import scipy.optimize
import scipy.special
from pyscf import lib

from momentGW import mpi_helper, util
from momentGW.pbc.tda import dTDA
from momentGW.rpa import dRPA as MoldRPA

# TODO: Check lack of Lai in the integrals


class dRPA(MoldRPA, dTDA):
    """
    Compute the self-energy moments using dRPA and numerical integration
    with periodic boundary conditions.

    Parameters
    ----------
    gw : BaseKGW
        GW object.
    nmom_max : int
        Maximum moment number to calculate.
    Lpx : numpy.ndarray
        Density-fitted ERI tensor, where the first two indices
        enumerate the k-points (as a dict), the third index is the auxiliary
        basis function index, and the fourth and fifth indices are
        the MO and Green's function orbital indices, respectively.
    integrals : KIntegrals
        Density-fitted integrals.
    mo_energy : dict, optional
        Molecular orbital energies. Keys are "g" and "w" for the Green's
        function and screened Coulomb interaction, respectively.
        If `None`, use `gw.mo_energy` for both. Default value is `None`.
    mo_occ : dict, optional
        Molecular orbital occupancies. Keys are "g" and "w" for the
        Green's function and screened Coulomb interaction, respectively.
        If `None`, use `gw.mo_occ` for both. Default value is `None`.
    """

    def __init__(
        self,
        gw,
        nmom_max,
        integrals,
        mo_energy=None,
        mo_occ=None,
    ):
        """
            Key quantities for the dRPA calculation. These are constructed in advance to
            reduce the computational cost of the quadratures.

            returns
            ----------
            Liaw : dict of numpy.ndarray
                The (Nkpt, Nkpt)(aux, W occ, W vir) integral array. The first Nkpt is the difference
                between k-points, the second is an index over the variations in this difference.
            diag_eri : numpy.ndarray
                Diagonal of the ERIs for each k-point.
            d : numpy.ndarray
                Array of orbital energy differences for each k-point.
            Liad : dict of numpy.ndarray
                The (Nkpt, Nkpt)(aux, W occ, W vir) integral array multiplied by the orbital energy
                differences.
            Liadinv,Laidinv : dict of numpy.ndarray
                The (Nkpt, Nkpt)(aux, W occ, W vir) integral array divided by the orbital energy
                differences.
            """
        #TODO discuss with Ollie if this really is optimal or will memory be too much of an issue.
        super().__init__(gw, nmom_max, integrals, mo_energy, mo_occ)

        kpts = self.kpts
        Lia = self.integrals.Lia

        self.Liaw = np.zeros((self.nkpts, self.nkpts), dtype=object)
        # self.Laiw = np.zeros((self.nkpts, self.nkpts), dtype=object)
        self.diag_eri = np.zeros((self.nkpts, self.nkpts), dtype=object)

        self.d = np.zeros((self.nkpts, self.nkpts), dtype=object)
        self.Liad = np.zeros((self.nkpts, self.nkpts), dtype=object)
        self.Laidinv = np.zeros((self.nkpts, self.nkpts), dtype=object)
        self.Liadinv = np.zeros((self.nkpts, self.nkpts), dtype=object)

        for q in kpts.loop(1):
            for kj in kpts.loop(1, mpi=True):
                kb = kpts.member(kpts.wrap_around(kpts[q] - kpts[kj]))
                self.Liaw[q, kb] = Lia[kj, kb]
                # self.Laiw[q, kb] = self.integrals.Lai[kj, kb] Lai?
                self.diag_eri[q, kb] = (
                                          lib.einsum(
                                              "np,np->p", Lia[kj, kb].conj(), Lia[kj, kb]
                                          )
                                          + lib.einsum(
                                      "np,np->p", Lia[kj, kb].conj(), Lia[kj, kb]
                                  )
                                  ) / (2 * self.nkpts) # Lai?

                self.d[q, kb] = util.build_1h1p_energies(
                    (self.mo_energy_w[kj], self.mo_energy_w[kb]),
                    (self.mo_occ_w[kj], self.mo_occ_w[kb]),
                ).ravel()

                self.Liad[q, kb] = Lia[kj, kb] * self.d[q, kb]
                self.Laidinv[q, kb] = (
                        Lia[kj, kb] / self.d[q, kb]
                )  # Lai?
                self.Liadinv[q, kb] = Lia[kj, kb] / self.d[q, kb]


    def integrate(self):
        """Optimise the quadrature and perform the integration for a
        given set of k points.

        Returns
        -------
        integral : numpy.ndarray
            Integral array, including the offset part.
        """

        cput0 = (lib.logger.process_clock(), lib.logger.perf_counter())
        lib.logger.info(self.gw, "Performing integration")
        lib.logger.debug(self.gw, "Memory usage: %.2f GB", self._memory_usage())

        # Get the offset integral quadrature
        quad = self.optimise_offset_quad()
        cput1 = lib.logger.timer(self.gw, "optimising offset quadrature", *cput0)

        # Perform the offset integral
        offset = self.eval_offset_integral(quad)
        cput1 = lib.logger.timer(self.gw, "performing offset integral", *cput1)

        # Get the main integral quadrature
        quad = self.optimise_main_quad()
        cput1 = lib.logger.timer(self.gw, "optimising main quadrature", *cput1)

        # Perform the main integral
        integral = self.eval_main_integral(quad)
        cput1 = lib.logger.timer(self.gw, "performing main integral", *cput1)

        # Report quadrature error
        if self.report_quadrature_error:
            a = 0.0
            b = 0.0
            for q in self.kpts.loop(1):
                for ka in self.kpts.loop(1, mpi=True):
                    a += np.sum((integral[0,q,ka] - integral[2,q,ka]) ** 2)
                    b += np.sum((integral[0,q,ka] - integral[1,q,ka]) ** 2)
            a, b = mpi_helper.allreduce(np.array([a, b]))
            a, b = a ** 0.5, b ** 0.5
            err = self.estimate_error_clencur(a, b)
            lib.logger.debug(self.gw, "One-quarter quadrature error: %s", a)
            lib.logger.debug(self.gw, "One-half quadrature error: %s", b)
            lib.logger.debug(self.gw, "Error estimate: %s", err)

        return integral[0] + offset

    def build_dd_moments(self, integral=None):
        """Build the moments of the density-density response.

        Parameters
        ----------
        integral : numpy.ndarray, optional
            Integral array, including the offset part. If `None`,
            calculate from scratch. Default is `None`.

        Variables
        ----------
        d : numpy.ndarray
            Array of orbital energy differences for each k-point.
        Liaw : dict of numpy.ndarray
            The (Nkpt, Nkpt)(aux, W occ, W vir) integral array. The first Nkpt is the difference
            between k-points, the second is an index over the variations in this difference.
        Liad : dict of numpy.ndarray
            The (Nkpt, Nkpt)(aux, W occ, W vir) integral array multiplied by the orbital energy
            differences. See "__init__" for more details.
        Liadinv,Laidinv : dict of numpy.ndarray
            The (Nkpt, Nkpt)(aux, W occ, W vir) integral array divided by the orbital energy
            differences. See "__init__" for more details.

        Returns
        -------
        moments : numpy.ndarray
            Moments of the density-density response.
        """

        cput0 = (lib.logger.process_clock(), lib.logger.perf_counter())
        lib.logger.info(self.gw, "Building density-density moments")
        lib.logger.debug(self.gw, "Memory usage: %.2f GB", self._memory_usage())

        kpts = self.kpts
        if integral is None:
            integral = self.integrate()

        moments = np.zeros((self.nkpts, self.nkpts, self.nmom_max + 1), dtype=object)

        for q in kpts.loop(1):
            # Get the zeroth order moment
            tmp = np.zeros((self.naux[q], self.naux[q]), dtype=complex)
            inter = 0.0
            for ka in kpts.loop(1, mpi=True):
                tmp += np.dot(self.Laidinv[q, ka], self.Liaw[q, ka].conj().T)
                inter += np.dot(integral[q, ka], self.Liadinv[q, ka].T.conj())
            tmp = mpi_helper.allreduce(tmp)
            tmp *= 2.0
            u = np.linalg.inv(np.eye(tmp.shape[0]) * self.nkpts / 2 + tmp)

            rest = np.dot(inter, u) * self.nkpts / 2
            for ka in kpts.loop(1, mpi=True):
                moments[q, ka, 0] = integral[q, ka] / self.d[q, ka] * self.nkpts / 2
                moments[q, ka, 0] -= np.dot(rest, self.Laidinv[q, ka]) * 2
        cput1 = lib.logger.timer(self.gw, "zeroth moment", *cput0)

        # Get the first order moment
        moments[:, :, 1] = self.Liad / self.nkpts
        cput1 = lib.logger.timer(self.gw, "first moment", *cput1)

        # Get the higher order moments
        for i in range(2, self.nmom_max + 1):
            for q in kpts.loop(1):
                tmp = 0.0
                for kb in kpts.loop(1, mpi=True):
                    moments[q, kb, i] = moments[q, kb, i - 2] * self.d[q, kb] ** 2
                    tmp += (
                        np.dot(moments[q, kb, i - 2], self.Liaw[q, kb].conj().T)
                        * 2
                        / self.nkpts
                    )  # aux^2 o v
                tmp = mpi_helper.allreduce(tmp)
                for ka in kpts.loop(1, mpi=True):
                    moments[q, ka, i] += np.dot(tmp, self.Liad[q, ka]) * 2  # aux^2 o v
            cput1 = lib.logger.timer(self.gw, "moment %d" % i, *cput1)

        return moments

    def optimise_offset_quad(self):
        """
        Optimise the grid spacing of Clenshaw-Curtis quadrature for the
        main integral.

        Variables
        ----------
        d : numpy.ndarray
            Array of orbital energy differences for each k-point.
        diag_eri : numpy.ndarray
            Diagonal of the ERIs for each k-point.

        Returns
        -------
        points : numpy.ndarray
            The quadrature points.
        weights : numpy.ndarrayf
            The quadrature weights.
        """

        bare_quad = self.gen_gausslag_quad_semiinf()
        exact = 0.0
        for q in self.kpts.loop(1):
            for ki in self.kpts.loop(1, mpi=True):
                exact += np.dot(1.0 / self.d[q, ki], self.d[q, ki] * self.diag_eri[q, ki])
        integrand = lambda quad: self.eval_diag_offset_integral(quad)
        quad = self.get_optimal_quad(bare_quad, integrand, exact)
        return quad

    def eval_diag_offset_integral(self, quad):
        """Evaluate the diagonal of the offset integral.

        Parameters
        ----------
        quad : tuple
            The quadrature points and weights.

        Variables
        ----------
        d : numpy.ndarray
            Array of orbital energy differences for each k-point.
        diag_eri : numpy.ndarray
            Diagonal of the ERIs for each k-point.

        Returns
        -------
        integral : numpy.ndarray
            Offset integral.
        """

        # TODO check this: is this still right, does it need a factor 4.0?

        integral = 0.0

        for point, weight in zip(*quad):
            for q in self.kpts.loop(1):
                for ka in self.kpts.loop(1, mpi=True):
                    tmp = self.d[q, ka] * self.diag_eri[q, ka]
                    expval = np.exp(-2 * point * self.d[q, ka])
                    res = np.dot(expval, tmp)
                    integral += 2 * res * weight
        return integral

    def eval_offset_integral(self, quad):
        """Evaluate the offset integral.

        Parameters
        ----------
        quad : tuple
            The quadrature points and weights.

        Variables
        ----------
        d : numpy.ndarray
            Array of orbital energy differences for each k-point.
        Liaw : dict of numpy.ndarray
            The (Nkpt, Nkpt)(aux, W occ, W vir) integral array. The first Nkpt is the difference
            between k-points, the second is an index over the variations in this difference.
        Liad : dict of numpy.ndarray
            The (Nkpt, Nkpt)(aux, W occ, W vir) integral array multiplied by the orbital energy
            differences. See "__init__" for more details.

        Returns
        -------
        integral : numpy.ndarray
            Offset integral.
        """
        integrals = 2 * self.Liad / (self.nkpts ** 2)

        kpts = self.kpts

        for point, weight in zip(*quad):
            rhs = np.zeros_like(integrals)
            for q in kpts.loop(1):
                lhs = 0.0
                for kb in kpts.loop(1, mpi=True):
                    lhs += np.dot(self.Liad[q, kb] * np.exp(-point * self.d[q, kb]), self.Liaw[q, kb].T.conj())
                lhs /= self.nkpts

                for kb in kpts.loop(1, mpi=True):
                    rhs[q, kb] = self.Liaw[q, kb] * np.exp(-point * self.d[q, kb]) + self.Liaw[
                        q, kb
                    ] * np.exp(
                        -point * self.d[q, kb]
                    )  # aux o v#(Lia[kj,kb]*np.exp(-point * d[q,kb]) + self.integrals.Lai[kj,kb]*np.exp(-point * d[q,kb]))/(2/self.nkpts) # aux o v
                    rhs[q, kb] /= self.nkpts**2
                    rhs[q, kb] = np.dot(lhs, rhs[q, kb])
                    integrals[q, kb] += rhs[q, kb] * weight * 4


            del rhs, lhs

        return integrals

    def optimise_main_quad(self):
        """
        Optimise the grid spacing of Clenshaw-Curtis quadrature for the
        main integral.

        Variables
        ----------
        d : numpy.ndarray
            Array of orbital energy differences for each k-point.
        diag_eri : numpy.ndarray
            Diagonal of the ERIs for each k-point.

        Returns
        -------
        points : numpy.ndarray
            The quadrature points.
        weights : numpy.ndarray
            The quadrature weights.
        """

        bare_quad = self.gen_clencur_quad_inf(even=True)

        exact = 0.0
        d_sq = np.zeros((self.nkpts, self.nkpts), dtype=object)
        d_eri = np.zeros((self.nkpts, self.nkpts), dtype=object)
        d_sq_eri = np.zeros((self.nkpts, self.nkpts), dtype=object)
        for q in self.kpts.loop(1):
            for kb in self.kpts.loop(1, mpi=True):
                exact += np.sum((self.d[q, kb] * (self.d[q, kb] + self.diag_eri[q, kb])) ** 0.5)
                exact -= 0.5 * np.dot(1.0 / self.d[q, kb], self.d[q, kb] * self.diag_eri[q, kb])
                exact -= np.sum(self.d[q, kb])
                d_sq[q, kb] = self.d[q, kb] ** 2
                d_eri[q, kb] = self.d[q, kb] * self.diag_eri[q, kb]
                d_sq_eri[q, kb] = self.d[q, kb] * (self.d[q, kb] + self.diag_eri[q, kb])

        integrand = lambda quad: self.eval_diag_main_integral(quad, d_sq, d_eri, d_sq_eri)
        quad = self.get_optimal_quad(bare_quad, integrand, exact)
        return quad

    def eval_diag_main_integral(self, quad, d_sq, d_eri, d_sq_eri):
        """Evaluate the diagonal of the main integral.

        Parameters
        ----------
        quad : tuple
            The quadrature points and weights.
        d_sq : numpy.ndarray
            Array of orbital energy differences squared for each k-point. See
            "optimise_main_quad" for more details.
        d_eri : numpy.ndarray
            Array of orbital energy differences times the diagonal of the ERIs for each
            k-point. See "optimise_main_quad" for more details.
        d_sq_eri : numpy.ndarray
            Array of orbital energy differences times the diagonal of the ERIs plus the orbital
            energy differences for each k-point. See "optimise_main_quad" for more details.

        Variables
        ----------
        d : numpy.ndarray
            Array of orbital energy differences for each k-point.
        diag_eri : numpy.ndarray
            Diagonal of the ERIs for each k-point.

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


        for point, weight in zip(*quad):
            contrib = 0.0
            for q in self.kpts.loop(1):
                for kb in self.kpts.loop(1, mpi=True):
                    contrib_int = diag_contrib(d_sq_eri[q, kb], point)
                    contrib_int -= diag_contrib(d_sq[q, kb], point)
                    contrib += np.abs(np.sum(contrib_int))

                    f_sq = 1.0 / (self.d[q, kb] ** 2 + point**2)** 2
                    contrib -= np.abs(point**2 * np.dot(f_sq, d_eri[q, kb]) / np.pi)
            integral += weight * contrib

        return integral

    def eval_main_integral(self, quad):
        """Evaluate the main integral.

        Parameters
        ----------
        quad : tuple
            The quadrature points and weights.

        Variables
        ----------
        d : numpy.ndarray
            Array of orbital energy differences for each k-point.
        Liaw : dict of numpy.ndarray
            The (Nkpt, Nkpt)(aux, W occ, W vir) integral array. The first Nkpt is the difference
            between k-points, the second is an index over the variations in this difference.
        Liad : dict of numpy.ndarray
            The (Nkpt, Nkpt)(aux, W occ, W vir) integral array multiplied by the orbital energy
            differences. See "__init__" for more details.

        Returns
        -------
        integral : numpy.ndarray
            Offset integral.
        """
        dim = 3 if self.report_quadrature_error else 1
        integral = np.zeros((3, self.nkpts, self.nkpts), dtype=object)
        kpts = self.kpts
        for i, (point, weight) in enumerate(zip(*quad)):
            contrib = np.zeros_like(self.d, dtype=object)

            for q in kpts.loop(1):
                f = np.zeros((self.nkpts), dtype=object)
                qz = 0.0
                for kb in kpts.loop(1, mpi=True):
                    f[kb] = 1.0 / (self.d[q, kb] ** 2 + point**2)
                    pre = (self.Liaw[q, kb] * f[kb] + self.Liaw[q, kb] * f[kb]) * (
                        2 / self.nkpts
                    )  # (Lia[kj, kb]*f[q,kb] + self.integrals.Lai[kj, kb]*f[q,kb])*(2/self.nkpts)
                    qz += np.dot(pre, self.Liad[q, kb].T.conj())
                qz = mpi_helper.allreduce(qz)

                tmp = np.linalg.inv(np.eye(self.naux[q]) + qz) - np.eye(self.naux[q])
                inner = np.dot(qz, tmp)

                for ka in kpts.loop(1, mpi=True):
                    contrib[q, ka] = 2 * np.dot(inner, self.Liaw[q, ka]) / (self.nkpts**2)
                    value = weight * (contrib[q, ka] * f[ka] * (point**2 / np.pi))

                    integral[0, q, ka] += value
                    if i % 2 == 0 and self.report_quadrature_error:
                        integral[1, q, ka] += 2 * value
                    if i % 4 == 0 and self.report_quadrature_error:
                        integral[2, q, ka] += 4 * value

        return integral

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
