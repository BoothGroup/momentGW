"""Construct RPA moments with unrestricted references."""

import numpy as np

from momentGW import logging, mpi_helper, util
from momentGW.rpa import dRPA as RdRPA
from momentGW.uhf.tda import dTDA


class dRPA(dTDA, RdRPA):
    """Compute the self-energy moments using dRPA and numerical integration
    with unrestricted references.

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
    def integrate(self):
        """Optimise the quadrature and perform the integration.

        Returns
        -------
        integrals: : tuple of numpy.ndarray
            Integral array, include the offset part, for each spin
            channel.
        """

        a0, a1 = self.mpi_slice(self.nov[0])
        b0, b1 = self.mpi_slice(self.nov[1])

        # Construct
        d_full = (
            util.build_1h1p_energies(self.mo_energy_w[0], self.mo_occ_w[0]).ravel(),
            util.build_1h1p_energies(self.mo_energy_w[1], self.mo_occ_w[1]).ravel(),
        )
        d = (d_full[0][a0:a1], d_full[1][b0:b1])

        # Calculate diagonal part of ERI
        diag_eri_α = np.zeros((self.nov[0],))
        diag_eri_α[a0:a1] = util.einsum("np,np->p", self.integrals[0].Lia, self.integrals[0].Lia)
        diag_eri_α = mpi_helper.allreduce(diag_eri_α)
        diag_eri_β = np.zeros((self.nov[1],))
        diag_eri_β[b0:b1] = util.einsum("np,np->p", self.integrals[1].Lia, self.integrals[1].Lia)
        diag_eri_β = mpi_helper.allreduce(diag_eri_β)
        diag_eri = (diag_eri_α, diag_eri_β)

        # Get the offset integral quadrature
        quad = (
            self.optimise_offset_quad(d_full[0], diag_eri[0], name="Offset (α)"),
            self.optimise_offset_quad(d_full[1], diag_eri[1], name="Offset (β)"),
        )

        # Perform the offset integral
        offset = (
            self.eval_offset_integral(quad[0], d[0], Lia=self.integrals[0].Lia),
            self.eval_offset_integral(quad[1], d[1], Lia=self.integrals[1].Lia),
        )

        # Get the main integral quadrature
        quad = (
            self.optimise_main_quad(d_full[0], diag_eri[0], name="Main (α)"),
            self.optimise_main_quad(d_full[1], diag_eri[1], name="Main (β)"),
        )

        # Perform the main integral
        integral = (
            self.eval_main_integral(quad[0], d[0], Lia=self.integrals[0].Lia),
            self.eval_main_integral(quad[1], d[1], Lia=self.integrals[1].Lia),
        )

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

        return (integral[0][0] + offset[0], integral[1][0] + offset[1])

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

        if integral is None:
            integral = self.integrate()

        a0, a1 = self.mpi_slice(self.nov[0])
        b0, b1 = self.mpi_slice(self.nov[1])
        moments = np.zeros((self.nmom_max + 1, self.naux, (a1 - a0) + (b1 - b0)))

        # Construct energy differences
        d = np.concatenate(
            [
                util.build_1h1p_energies(self.mo_energy_w[0], self.mo_occ_w[0]).ravel()[a0:a1],
                util.build_1h1p_energies(self.mo_energy_w[1], self.mo_occ_w[1]).ravel()[b0:b1],
            ]
        )

        # Calculate (L|ia) D_{ia} and (L|ia) D_{ia}^{-1} intermediates
        Lia = np.concatenate(
            [
                self.integrals[0].Lia,
                self.integrals[1].Lia,
            ],
            axis=1,
        )
        Liad = Lia * d[None]
        Liadinv = Lia / d[None]
        integral = np.concatenate(integral, axis=1)

        # Construct (A-B)^{-1}
        u = np.dot(Liadinv, Lia.T) * 2.0
        u = mpi_helper.allreduce(u)
        u = np.linalg.inv(np.eye(self.naux) + u)

        # Get the zeroth order moment
        moments[0] = integral / d[None]
        tmp = np.linalg.multi_dot((integral, Liadinv.T, u))
        tmp = mpi_helper.allreduce(tmp)
        moments[0] -= np.dot(tmp, Liadinv) * 2.0
        del u, tmp

        # Get the first orer moment
        moments[1] = Liad

        # Get the higher order moments
        for i in range(2, self.nmom_max + 1):
            moments[i] = moments[i - 2] * d[None] ** 2
            tmp = np.dot(moments[i - 2], Lia.T)
            tmp = mpi_helper.allreduce(tmp)
            moments[i] += np.dot(tmp, Liad) * 2.0
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
