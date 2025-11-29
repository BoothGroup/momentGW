"""Construct RPA moments with unrestricted references."""

import numpy as np

from momentGW import logging, mpi_helper, util
from momentGW.rpa import dRPA as RdRPA
from momentGW.uhf.tda import dTDA


class dRPA(dTDA, RdRPA):
    """Compute the self-energy moments using dRPA and numerical integration with unrestricted
    references.

    Parameters
    ----------
    gw : BaseUGW
        GW object.
    nmom_max : int
        Maximum moment number to calculate.
    integrals : UIntegrals
        Integrals object.
    mo_energy : dict, optional
        Molecular orbital energies for each spin. Keys are "g" and "w"
        for the Green's function and screened Coulomb interaction,
        respectively. If `None`, use `gw.mo_energy` for both. Default
        value is `None`.
    mo_occ : dict, optional
        Molecular orbital occupancies for each spin. Keys are "g" and
        "w" for the Green's function and screened Coulomb interaction,
        respectively. If `None`, use `gw.mo_occ` for both. Default
        value is `None`.
    """

    @logging.with_timer("Numerical integration")
    @logging.with_status("Performing numerical integration")
    def build_zeroth_moment(self):
        """Optimise the quadrature and perform the integration.

        Returns
        -------
        integrals: : tuple of numpy.ndarray
            Integral array, include the offset part, for each spin
            channel.
        """

        a0, a1 = self.mpi_slice(self.nov[0])
        b0, b1 = self.mpi_slice(self.nov[1])

        # Construct d
        d = np.concatenate(
            [
                util.build_1h1p_energies(self.mo_energy_w[0], self.mo_occ_w[0]).ravel()[a0:a1],
                util.build_1h1p_energies(self.mo_energy_w[1], self.mo_occ_w[1]).ravel()[b0:b1],
            ]
        )

        # Calculate diagonal part of ERI
        diag_eri_α = np.zeros((self.nov[0],))
        diag_eri_α[a0:a1] = util.einsum("np,np->p", self.integrals[0].Lia, self.integrals[0].Lia)
        diag_eri_α = mpi_helper.allreduce(diag_eri_α)
        diag_eri_β = np.zeros((self.nov[1],))
        diag_eri_β[b0:b1] = util.einsum("np,np->p", self.integrals[1].Lia, self.integrals[1].Lia)
        diag_eri_β = mpi_helper.allreduce(diag_eri_β)
        diag_eri = np.concatenate([diag_eri_α, diag_eri_β])

        # Calculate (L|ia) D_{ia} and (L|ia) D_{ia}^{-1} intermediates
        Lia = np.concatenate(
            [
                self.integrals[0].Lia,
                self.integrals[1].Lia,
            ],
            axis=1,
        )

        quad = self.optimise_main_quad(d, diag_eri, name="Combined ERI")
        integral = self.eval_main_integral(quad, d, Lia=Lia, spin=True)

        # Report quadrature errors
        if self.report_quadrature_error:
            a = (
                np.sum((integral[0][0] - integral[0][2]) ** 2),
                np.sum((integral[1][0] - integral[1][2]) ** 2),
            )
            b = (
                np.sum((integral[0][0] - integral[0][1]) ** 2),
                np.sum((integral[1][0] - integral[1][1]) ** 2),
            )
            a, b = mpi_helper.allreduce(np.array([a, b]))
            a, b = a**0.5, b**0.5
            err = (self.estimate_error_clencur(a[0], b[0]), self.estimate_error_clencur(a[1], b[1]))
            for s, spin in enumerate(["α", "β"]):
                style_half = logging.rate(a[s], 1e-4, 1e-3)
                style_quar = logging.rate(b[s], 1e-8, 1e-6)
                style_full = logging.rate(err[s], 1e-12, 1e-9)
                logging.write(
                    f"Error in integral ({spin}):  [{style_full}]{err[s]:.3e}[/] "
                    f"(half = [{style_half}]{a[s]:.3e}[/], quarter = [{style_quar}]{b[s]:.3e}[/])",
                )

        return integral[0]

    @logging.with_timer("Nth density-density moments")
    @logging.with_status("Constructing nth density-density moment")
    def build_nth_dd_moment(self, n, recursion_term=None, zeroth_mom=None, Lia=None):
        """Build the nth moment of the density-density response.

        Parameters
        ----------
        n : int
            Moment order to be built.
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
        if Lia is None:
            Lia = np.concatenate([self.integrals[0].Lia, self.integrals[1].Lia], axis=1)

        if n % 2 == 0:
            if zeroth_mom is None:
                zeroth_mom = self.build_zeroth_moment()
            if n != 0:
                tmp = np.dot(Lia * self.d[None], recursion_term) * 2.0
                tmp = mpi_helper.allreduce(tmp)
                recursion_term = util.einsum("i, iP->iP", self.d**2, recursion_term)
                recursion_term += util.einsum("Pi,PQ->iQ", Lia, tmp)
                del tmp
            elif n == 0 and recursion_term is None:
                recursion_term = Lia.T
            return recursion_term, np.dot(zeroth_mom, recursion_term)

        else:
            if recursion_term is None:
                raise AttributeError(
                    f"To build the {n}th dd-moment, a recursion_term must be provided"
                )
            return recursion_term, np.dot(Lia * self.d[None], recursion_term)

    @logging.with_timer("Density-density moments")
    @logging.with_status("Constructing density-density moments")
    def build_dd_moments(self, integral=None):
        """Build the moments of the density-density response.

        Parameters
        ----------
        integral : tuple of numpy.ndarray, optional
            Integral array, include the offset part, for each spin
            channel. If `None`, calculate from scratch. Default value is
            `None`.

        Returns
        -------
        moments : tuple of numpy.ndarray
            Moments of the density-density response.
        """

        # Construct energy differences
        if self.d is None:
            self._build_d()

        if integral is None:
            integral = self.build_zeroth_moment()

        a0, a1 = self.mpi_slice(self.nov[0])
        b0, b1 = self.mpi_slice(self.nov[1])
        moments = np.zeros((self.nmom_max + 1, self.naux, (a1 - a0) + (b1 - b0)))

        # Calculate (L|ia) D_{ia} and (L|ia) D_{ia}^{-1} intermediates
        Lia = np.concatenate(
            [
                self.integrals[0].Lia,
                self.integrals[1].Lia,
            ],
            axis=1,
        )

        moments[0] = integral

        # Get the first order moment
        moments[1] = Lia * self.d[None]

        # Get the higher order moments
        for i in range(2, self.nmom_max + 1):
            moments[i] = moments[i - 2] * self.d[None] ** 2
            tmp = np.dot(moments[i - 2], Lia.T)
            tmp = mpi_helper.allreduce(tmp)
            moments[i] += np.dot(tmp, moments[1]) * 2.0
            del tmp

        return moments

    @logging.with_timer("Density-density moments")
    @logging.with_status("Constructing density-density moments")
    def build_dd_moments_exact(self):
        """Build the exact moments of the density-density response.

        Notes
        -----
        Placeholder for future implementation.
        """
        raise NotImplementedError
