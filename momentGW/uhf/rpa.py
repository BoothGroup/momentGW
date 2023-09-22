"""
Construct RPA moments with unrestricted references.
"""

import numpy as np
from pyscf import lib
from pyscf.agf2 import mpi_helper

from momentGW.rpa import RPA as RRPA
from momentGW.uhf.tda import TDA


class RPA(TDA, RRPA):
    """
    Compute the self-energy moments using dRPA and numerical integration
    with unrestricted references.

    Parameters
    ----------
    gw : BaseUGW
        GW object.
    nmom_max : int
        Maximum moment number to calculate.
    integrals : UIntegrals
        Density-fitted integrals.
    mo_energy : tuple of (numpy.ndarray or tuple of numpy.ndarray), optional
        Molecular orbital energies for each spin channel. If either
        element is a tuple, the first element corresponds to the Green's
        function basis and the second to the screened Coulomb
        interaction. Default value is that of `gw.mo_energy`.
    mo_occ : tuple of (numpy.ndarray or tuple of numpy.ndarray), optional
        Molecular orbital occupancies for each spin channel. If either
        element is a tuple, the first element corresponds to the Green's
        function basis and the second to the screened Coulomb
        interaction. Default value is that of `gw.mo_occ`.
    """

    def integrate(self):
        """Optimise the quadrature and perform the integration.

        Returns
        -------
        integrals: : tuple of numpy.ndarray
            Integral array, include the offset part, for each spin
            channel.
        """

        cput0 = (lib.logger.process_clock(), lib.logger.perf_counter())
        lib.logger.info(self.gw, "Performing integration")
        lib.logger.debug(self.gw, "Memory usage: %.2f GB", self._memory_usage())
        a0, a1 = self.mpi_slice(self.nov[0])
        b0, b1 = self.mpi_slice(self.nov[1])

        # Construct
        d_full = (
            lib.direct_sum(
                "a-i->ia",
                self.mo_energy_w[0][self.mo_occ_w[0] == 0],
                self.mo_energy_w[0][self.mo_occ_w[0] > 0],
            ).ravel(),
            lib.direct_sum(
                "a-i->ia",
                self.mo_energy_w[1][self.mo_occ_w[1] == 0],
                self.mo_energy_w[1][self.mo_occ_w[1] > 0],
            ).ravel(),
        )
        d = (d_full[0][a0:a1], d_full[1][b0:b1])

        # Calculate diagonal part of ERI
        diag_eri_α = np.zeros((self.nov[0],))
        diag_eri_α[a0:a1] = lib.einsum("np,np->p", self.integrals[0].Lia, self.integrals[0].Lia)
        diag_eri_α = mpi_helper.allreduce(diag_eri_α)
        diag_eri_β = np.zeros((self.nov[1],))
        diag_eri_β[b0:b1] = lib.einsum("np,np->p", self.integrals[1].Lia, self.integrals[1].Lia)
        diag_eri_β = mpi_helper.allreduce(diag_eri_β)
        diag_eri = (diag_eri_α, diag_eri_β)

        # Get the offset integral quadrature
        quad = (
            self.optimise_offset_quad(d_full[0], diag_eri[0]),
            self.optimise_offset_quad(d_full[1], diag_eri[1]),
        )
        cput1 = lib.logger.timer(self.gw, "optimising offset quadrature", *cput0)

        # Perform the offset integral
        # FIXME do these offset integrals need a sum over spin?
        offset = (
            self.eval_offset_integral(quad[0], d[0], Lia=self.integrals[0].Lia),
            self.eval_offset_integral(quad[1], d[1], Lia=self.integrals[1].Lia),
        )
        cput1 = lib.logger.timer(self.gw, "performing offset integral", *cput1)

        # Get the main integral quadrature
        quad = (
            self.optimise_main_quad(d_full[0], diag_eri[0]),
            self.optimise_main_quad(d_full[1], diag_eri[1]),
        )
        cput1 = lib.logger.timer(self.gw, "optimising main quadrature", *cput1)

        # Perform the main integral
        integral = (
            self.eval_main_integral(quad[0], d[0], Lia=self.integrals[0].Lia),
            self.eval_main_integral(quad[1], d[1], Lia=self.integrals[1].Lia),
        )
        cput1 = lib.logger.timer(self.gw, "performing main integral", *cput1)

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
            lib.logger.debug(self.gw, "One-quarter quadrature error: %s", a)
            lib.logger.debug(self.gw, "One-half quadrature error: %s", b)
            lib.logger.debug(self.gw, "Error estimate: %s", err)

        return (integral[0][0] + offset[0], integral[1][0] + offset[1])

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

        cput0 = (lib.logger.process_clock(), lib.logger.perf_counter())
        lib.logger.info(self.gw, "Building density-density moments")
        lib.logger.debug(self.gw, "Memory usage: %.2f GB", self._memory_usage())

        assert self.naux[0] == self.naux[1]
        moments = np.zeros((self.nmom_max + 1, self.naux[0], self.nov[0] + self.nov[1]))

        # Construct energy differences
        d = np.concatenate(
            [
                lib.direct_sum(
                    "a-i->ia",
                    self.mo_energy_w[0][self.mo_occ_w[0] == 0],
                    self.mo_energy_w[0][self.mo_occ_w[0] > 0],
                ).ravel(),
                lib.direct_sum(
                    "a-i->ia",
                    self.mo_energy_w[1][self.mo_occ_w[1] == 0],
                    self.mo_energy_w[1][self.mo_occ_w[1] > 0],
                ).ravel(),
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
        u = np.linalg.inv(np.eye(self.naux[0]) + u)
        cput1 = lib.logger.timer(self.gw, "constructing (A-B)^{-1}", *cput0)

        # Get the zeroth order moment
        moments[0] = integral / d[None]
        tmp = np.linalg.multi_dot((integral, Liadinv.T, u))
        moments[0] -= np.dot(tmp, Liadinv) * 2.0
        del u, tmp
        cput1 = lib.logger.timer(self.gw, "zeroth moment", *cput1)

        # Get the first orer moment
        moments[1] = Liad

        # Get the higher order moments
        for i in range(2, self.nmom_max + 1):
            moments[i] = moments[i - 2] * d[None] ** 2
            tmp = np.dot(moments[i - 2], Lia.T)
            moments[i] += np.dot(tmp, Liad) * 2.0
            del tmp

            cput1 = lib.logger.timer(self.gw, "moment %d" % i, *cput1)

        return moments

    def build_dd_moments_exact(self):
        raise NotImplementedError
