"""
Spin-unrestricted quasiparticle self-consistent GW via self-energy
moment constraints for molecular systems.
"""

import numpy as np
from pyscf import lib
from pyscf.lib import logger

from momentGW import util
from momentGW.qsgw import qsGW
from momentGW.uhf import UGW


class qsUGW(UGW, qsGW):
    __doc__ = qsGW.__doc__.replace("Spin-restricted", "Spin-unrestricted", 1)

    # --- Default qsUGW options

    solver = UGW

    _opts = util.list_union(UGW._opts, qsGW._opts)

    @property
    def name(self):
        """Method name."""
        polarizability = self.polarizability.upper().replace("DTDA", "dTDA").replace("DRPA", "dRPA")
        return f"{polarizability}-qsUGW"

    @staticmethod
    def project_basis(matrix, ovlp, mo1, mo2):
        """
        Project a matrix from one basis to another.

        Parameters
        ----------
        matrix : numpy.ndarray or tuple of (GreensFunction or SelfEnergy)
            Matrix to project for each spin channel. Can also be a tuple
            of `GreensFunction` or `SelfEnergy` objects, in which case
            the `couplings` attributes are projected.
        ovlp : numpy.ndarray
            Overlap matrix in the shared (AO) basis.
        mo1 : numpy.ndarray
            First basis, rotates from the shared (AO) basis into the
            basis of `matrix` for each spin channel.
        mo2 : numpy.ndarray
            Second basis, rotates from the shared (AO) basis into the
            desired basis of the output for each spin channel.

        Returns
        -------
        projected_matrix : numpy.ndarray or tuple of (GreensFunction or SelfEnergy)
            Matrix projected into the desired basis for each spin
            channel.
        """

        proj = lib.einsum("pq,spi,sqj->sij", ovlp, np.conj(mo1), mo2)

        if isinstance(matrix, np.ndarray):
            projected_matrix = lib.einsum(
                "...pq,s...pi,s...qj->s...ij", matrix, np.conj(proj), proj
            )
        else:
            projected_matrix = []
            for s, m in enumerate(matrix):
                coupling = lib.einsum("pk,pi->ik", m.coupling, np.conj(proj[s]))
                projected_m = m.copy()
                projected_m.coupling = coupling
                projected_matrix.append(projected_m)
            projected_matrix = tuple(projected_matrix)

        return projected_matrix

    @staticmethod
    def self_energy_to_moments(se, nmom_max):
        """
        Return the hole and particle moments for a self-energy.

        Parameters
        ----------
        se : tuple of SelfEnergy
            Self-energy to compute the moments of for each spin channel.

        Returns
        -------
        th : numpy.ndarray
            Hole moments for each spin channel.
        tp : numpy.ndarray
            Particle moments for each spin channel.
        """
        th = np.array([s.get_occupied().moment(range(nmom_max + 1)) for s in se])
        tp = np.array([s.get_virtual().moment(range(nmom_max + 1)) for s in se])
        return th, tp

    def build_static_potential(self, mo_energy, se):
        """
        Build the static potential approximation to the self-energy.

        Parameters
        ----------
        mo_energy : numpy.ndarray
            Molecular orbital energies.
        se : SelfEnergy
            Self-energy to approximate.

        Returns
        -------
        se_qp : numpy.ndarray
            Static potential approximation to the self-energy.
        """
        return np.array([qsGW.build_static_potential(self, e, s) for e, s in zip(mo_energy, se)])

    # TODO move this to evUGW
    def check_convergence(self, mo_energy, mo_energy_prev, th, th_prev, tp, tp_prev):
        """Check for convergence, and print a summary of changes.

        Parameters
        ----------
        mo_energy : numpy.ndarray
            Molecular orbital energies for each spin channel.
        mo_energy_prev : numpy.ndarray
            Molecular orbital energies from the previous iteration for
            each spin channel.
        th : numpy.ndarray
            Moments of the occupied self-energy for each spin channel.
        th_prev : numpy.ndarray
            Moments of the occupied self-energy from the previous
            iteration for each spin channel.
        tp : numpy.ndarray
            Moments of the virtual self-energy for each spin channel.
        tp_prev : numpy.ndarray
            Moments of the virtual self-energy from the previous
            iteration for each spin channel.

        Returns
        -------
        conv : bool
            Convergence flag.
        """

        if th_prev is None:
            th_prev = np.zeros_like(th)
        if tp_prev is None:
            tp_prev = np.zeros_like(tp)

        error_homo = (
            abs(mo_energy[0][self.nocc[0] - 1] - mo_energy_prev[0][self.nocc[0] - 1]),
            abs(mo_energy[1][self.nocc[1] - 1] - mo_energy_prev[1][self.nocc[1] - 1]),
        )
        error_lumo = (
            abs(mo_energy[0][self.nocc[0]] - mo_energy_prev[0][self.nocc[0]]),
            abs(mo_energy[1][self.nocc[1]] - mo_energy_prev[1][self.nocc[1]]),
        )

        error_th = (self._moment_error(th[0], th_prev[0]), self._moment_error(th[1], th_prev[1]))
        error_tp = (self._moment_error(tp[0], tp_prev[0]), self._moment_error(tp[1], tp_prev[1]))

        logger.info(
            self, "Change in QPs (α): HOMO = %.6g  LUMO = %.6g", error_homo[0], error_lumo[0]
        )
        logger.info(
            self, "Change in QPs (β): HOMO = %.6g  LUMO = %.6g", error_homo[1], error_lumo[1]
        )
        logger.info(self, "Change in moments (α): occ = %.6g  vir = %.6g", error_th[0], error_tp[0])
        logger.info(self, "Change in moments (β): occ = %.6g  vir = %.6g", error_th[1], error_tp[1])

        return self.conv_logical(
            (
                max(max(error_homo), max(error_lumo)) < self.conv_tol,
                max(max(error_th), max(error_tp)) < self.conv_tol_moms,
            )
        )
