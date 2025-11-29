"""Construct RPA moments with periodic boundary conditions."""

import numpy as np

from momentGW import logging, mpi_helper, util
from momentGW.pbc.tda import dTDA
from momentGW.rpa import dRPA as MoldRPA


class dRPA(dTDA, MoldRPA):
    """Compute the self-energy moments using dRPA and numerical integration with periodic boundary
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

    @logging.with_timer("Numerical integration")
    @logging.with_status("Performing numerical integration")
    def build_zeroth_moment(self):
        """Optimise the quadrature and perform the integration for a given set of k-points for the
        zeroth moment.

        Returns
        -------
        integral : numpy.ndarray
            Integral array, including the offset part.
        """

        # Calculate diagonal part of ERIs
        diag_eri = self._build_diag_eri()

        # Get the main integral quadrature
        quad = self.optimise_main_quad(self.d, diag_eri)

        # Perform the main integral
        integral = self.eval_main_integral(quad, self.d)

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
            integral = np.delete(integral, [1, 2], 0)

        return integral[0]

    @logging.with_timer("Nth density-density moments")
    @logging.with_status("Constructing nth density-density moment")
    def build_nth_dd_moment(self, n, q, recursion_term=None, zeroth_mom=None):
        """Build the nth moment of the density-density response for a given set of k-points.

        Parameters
        ----------
        n : int
            Moment order to be built.
        q : int
            Index associated with a difference in k-points
        recursion_term : numpy.ndarray, optional
            Previous recursion term required to build the next moment. In the case of RPA this is
            the appropriate [(A+B)(A-B)]^(n-2/2) for the nth moment. These are only calculated on
            even moments, odd moments use the previous even moment value.
        zeroth moment : numpy.ndarray, optional
            Zeroth moment of the density-density response.

        Returns
        -------
        recursion_term : numpy.ndarray
            Term required for the next moment. In the case of RPA this is [(A+B)(A-B)]^(n/2)
        eta_aux : numpy.ndarray
            The nth density-density response moment in (N_aux,N_aux) form
        """
        kpts = self.kpts
        eta_aux = 0
        if n % 2 == 0:
            if zeroth_mom is None:
                raise AttributeError(
                    "0th moment must be provided by build_zeroth_moment for k-point calculations."
                )
            if n != 0:
                tmp = 0.0
                for ka in kpts.loop(1, mpi=True):
                    kb = kpts.member(kpts.wrap_around(kpts[q] + kpts[ka]))
                    tmp += np.dot(self.integrals.Lia[ka, kb] * self.d[q, kb], recursion_term[q, kb])
                tmp = mpi_helper.allreduce(tmp)
                tmp *= 4 / self.nkpts
                for ka in kpts.loop(1, mpi=True):
                    kb = kpts.member(kpts.wrap_around(kpts[q] + kpts[ka]))
                    recursion_term[q, kb] = util.einsum(
                        "i, iP->iP", self.d[q, kb] ** 2, recursion_term[q, kb]
                    )
                    recursion_term[q, kb] += util.einsum(
                        "Pi,PQ->iQ", self.integrals.Lia[ka, kb].conj(), tmp
                    )
                    eta_aux += np.dot(zeroth_mom[q, kb], recursion_term[q, kb])
            else:
                if recursion_term is None:
                    recursion_term = np.zeros_like(zeroth_mom)
                for ka in kpts.loop(1, mpi=True):
                    kb = kpts.member(kpts.wrap_around(kpts[q] + kpts[ka]))
                    recursion_term[q, kb] += self.integrals.Lia[ka, kb].T.conj()
                    eta_aux += np.dot(zeroth_mom[q, kb], recursion_term[q, kb])

        else:
            if recursion_term is None:
                raise AttributeError(
                    f"To build the {n}th dd-moment, a recursion_term must be provided"
                )
            for ka in kpts.loop(1, mpi=True):
                kb = kpts.member(kpts.wrap_around(kpts[q] + kpts[ka]))
                eta_aux += (
                    np.dot(self.integrals.Lia[ka, kb] * self.d[q, kb][None], recursion_term[q, kb])
                    / self.nkpts
                )

        eta_aux = mpi_helper.allreduce(eta_aux)
        eta_aux *= 2.0 / self.nkpts

        return recursion_term, eta_aux

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

        if self.d is None:
            self._build_d()

        if integral is None:
            integral = self.build_zeroth_moment()

        kpts = self.kpts
        Lia = self.integrals.Lia
        moments = np.zeros((self.nkpts, self.nkpts, self.nmom_max + 1), dtype=object)

        moments[:, :, 0] = integral

        # Get the first order moment
        for q in kpts.loop(1):
            for ki in self.kpts.loop(1, mpi=True):
                kb = self.kpts.member(self.kpts.wrap_around(self.kpts[q] + self.kpts[ki]))
                moments[q, kb, 1] = Lia[ki, kb] * self.d[q, kb]

        # Get the higher order moments
        for i in range(2, self.nmom_max + 1):
            for q in kpts.loop(1):
                tmp = 0.0
                for ka in kpts.loop(1, mpi=True):
                    kb = kpts.member(kpts.wrap_around(kpts[q] + kpts[ka]))
                    moments[q, kb, i] = moments[q, kb, i - 2] * self.d[q, kb] ** 2
                    tmp += np.dot(moments[q, kb, i - 2], self.integrals.Lia[ka, kb].conj().T)
                tmp = mpi_helper.allreduce(tmp)
                tmp *= 2 / self.nkpts
                for ki in kpts.loop(1, mpi=True):
                    ka = kpts.member(kpts.wrap_around(kpts[q] + kpts[ki]))
                    moments[q, ka, i] += np.dot(tmp, moments[q, ka, 1]) * 2

        moments[:, :, 1] /= self.nkpts  # Moment 1 used as Lia*d, needed to correct it for mom 1

        return moments

    def optimise_main_quad(self, d, diag_eri, name="main"):
        """Optimise the grid spacing of Clenshaw-Curtis quadrature for the main integral.

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
        bare_quad = self.gen_ClenCur_quad_semiinf()

        # Calculate the exact value of the integral for the diagonal
        exact = 0.0
        for q in self.kpts.loop(1):
            for kj in self.kpts.loop(1, mpi=True):
                kb = self.kpts.member(self.kpts.wrap_around(self.kpts[q] + self.kpts[kj]))
                exact += np.sum(d[q, kb] * ((d[q, kb] * (d[q, kb] + diag_eri[q, kb])) ** -0.5))
        exact = mpi_helper.allreduce(exact)

        # Define the integrand
        integrand = lambda quad: self.eval_diag_main_integral(quad, d, diag_eri)

        # Get the optimal quadrature
        quad = self.get_optimal_quad(bare_quad, integrand, exact, name=name)

        return quad

    def eval_diag_main_integral(self, quad, d, diag_eri):
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

        # Calculate the integral for each point
        for point, weight in zip(*quad):
            contrib = 0.0
            for q in self.kpts.loop(1):
                for kj in self.kpts.loop(1, mpi=True):
                    kb = self.kpts.member(self.kpts.wrap_around(self.kpts[q] + self.kpts[kj]))
                    val = (d[q, kb] + diag_eri[q, kb]) * d[q, kb] + point**2
                    contrib += np.sum(d[q, kb] * val ** (-1)) * 2 / np.pi
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

        # Initialise the integral
        dim = 3 if self.report_quadrature_error else 1
        integral = np.zeros((dim, self.nkpts, self.nkpts), dtype=object)

        # Calculate the integral for each point
        kpts = self.kpts
        naux = self.naux
        for i, (point, weight) in enumerate(zip(*quad)):
            for q in kpts.loop(1):
                f = np.zeros((self.nkpts), dtype=object)
                qz = 0.0
                for ki in kpts.loop(1, mpi=True):
                    kj = kpts.member(kpts.wrap_around(kpts[q] + kpts[ki]))
                    f[kj] = d[q, kj] / (d[q, kj] ** 2 + point**2)
                    pre = (Lia[ki, kj] * f[kj]) * (4 / self.nkpts)
                    qz += np.dot(pre, Lia[ki, kj].T.conj())
                qz = mpi_helper.allreduce(qz)

                tmp = np.linalg.inv(np.eye(naux[q]) + qz) - np.eye(naux[q])

                for ka in kpts.loop(1, mpi=True):
                    kb = kpts.member(kpts.wrap_around(kpts[q] + kpts[ka]))
                    if i == 0:
                        for v in range(dim):
                            integral[v, q, kb] += Lia[ka, kb] / self.nkpts

                    value = weight * np.dot(tmp, Lia[ka, kb] * f[kb]) * (2 / (np.pi * self.nkpts))

                    integral[0, q, kb] += value
                    if i % 2 == 0 and self.report_quadrature_error:
                        integral[1, q, kb] += 2 * value
                    if i % 4 == 0 and self.report_quadrature_error:
                        integral[2, q, kb] += 4 * value

        return integral
