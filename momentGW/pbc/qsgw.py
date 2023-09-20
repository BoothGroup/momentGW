"""
Spin-restricted quasiparticle self-consistent GW via self-energy moment
constraints for periodic systems.
"""

import numpy as np
from pyscf import lib

from momentGW import util
from momentGW.pbc.evgw import evKGW
from momentGW.pbc.gw import KGW
from momentGW.qsgw import qsGW


class qsKGW(KGW, qsGW):
    __doc__ = qsGW.__doc__.replace("molecules", "periodic systems", 1)

    # --- Default qsKGW options

    solver = KGW

    _opts = util.list_union(KGW._opts, qsGW._opts)

    @property
    def name(self):
        """Method name."""
        polarizability = self.polarizability.upper().replace("DTDA", "dTDA").replace("DRPA", "dRPA")
        return f"{polarizability}-qsKGW"

    @staticmethod
    def project_basis(matrix, ovlp, mo1, mo2):
        """
        Project a matrix from one basis to another.

        Parameters
        ----------
        matrix : numpy.ndarray or tuple of (GreensFunction or SelfEnergy)
            Matrix to project at each k-point. Can also be a tuple of
            `GreensFunction` or `SelfEnergy` objects, in which case the
            `couplings` attributes are projected.
        ovlp : numpy.ndarray
            Overlap matrix in the shared (AO) basis at each k-point.
        mo1 : numpy.ndarray
            First basis, rotates from the shared (AO) basis into the
            basis of `matrix` at each k-point.
        mo2 : numpy.ndarray
            Second basis, rotates from the shared (AO) basis into the
            desired basis of the output at each k-point.

        Returns
        -------
        projected_matrix : numpy.ndarray or tuple of (GreensFunction or SelfEnergy)
            Matrix projected into the desired basis at each k-point.
        """

        proj = lib.einsum("k...pq,k...pi,k...qj->k...ij", ovlp, np.conj(mo1), mo2)

        if isinstance(matrix, np.ndarray):
            projected_matrix = lib.einsum(
                "k...pq,k...pi,k...qj->k...ij", matrix, np.conj(proj), proj
            )
        else:
            projected_matrix = []
            for k, m in enumerate(matrix):
                coupling = lib.einsum("pk,pi->ik", m.coupling, np.conj(proj[k]))
                projected_m = m.copy()
                projected_m.coupling = coupling
                projected_matrix.append(projected_m)

        return projected_matrix

    @staticmethod
    def self_energy_to_moments(se, nmom_max):
        """
        Return the hole and particle moments for a self-energy.

        Parameters
        ----------
        se : tuple of SelfEnergy
            Self-energy to compute the moments of at each k-point.

        Returns
        -------
        th : numpy.ndarray
            Hole moments at each k-point.
        tp : numpy.ndarray
            Particle moments at each k-point.
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
            Molecular orbital energies at each k-point.
        se : tuple of SelfEnergy
            Self-energy to approximate at each k-point.

        Returns
        -------
        se_qp : numpy.ndarray
            Static potential approximation to the self-energy at each
            k-point.
        """
        return np.array([qsGW.build_static_potential(self, mo, s) for mo, s in zip(mo_energy, se)])

    check_convergence = evKGW.check_convergence
