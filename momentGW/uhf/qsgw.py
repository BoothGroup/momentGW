"""
Spin-unrestricted quasiparticle self-consistent GW via self-energy
moment constraints for molecular systems.
"""

import numpy as np
from pyscf import lib

from momentGW import util
from momentGW.qsgw import qsGW
from momentGW.uhf import UGW, evUGW


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

    check_convergence = evUGW.check_convergence
