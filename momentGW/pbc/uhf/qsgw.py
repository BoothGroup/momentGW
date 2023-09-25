"""
Spin-unrestricted quasiparticle self-consistent GW via self-energy
moment constraints for periodic systems.
"""

import numpy as np
from pyscf import lib

from momentGW import qsGW, util
from momentGW.pbc.qsgw import qsKGW
from momentGW.pbc.uhf.evgw import evKUGW
from momentGW.pbc.uhf.gw import KUGW
from momentGW.uhf.qsgw import qsUGW


class qsKUGW(KUGW, qsKGW, qsUGW):  # noqa: D101
    __doc__ = qsKGW.__doc__.replace("Spin-restricted", "Spin-unrestricted", 1)

    # --- Default qsKUGW options

    solver = KUGW

    _opts = util.list_union(KUGW._opts, qsKGW._opts, qsUGW._opts)

    @property
    def name(self):
        """Method name."""
        polarizability = self.polarizability.upper().replace("DTDA", "dTDA").replace("DRPA", "dRPA")
        return f"{polarizability}-qsKUGW"

    @staticmethod
    def project_basis(matrix, ovlp, mo1, mo2):
        """
        Project a matrix from one basis to another.

        Parameters
        ----------
        matrix : numpy.ndarray or tuple of dyson.Lehmann
            Matrix to project at each k-point for each spin channel. Can
            also be a tuple of `dyson.Lehmann` objects, in which case the
            `couplings` attributes are projected.
        ovlp : numpy.ndarray
            Overlap matrix in the shared (AO) basis at each k-point.
        mo1 : numpy.ndarray
            First basis, rotates from the shared (AO) basis into the
            basis of `matrix` at each k-point for each spin channel.
        mo2 : numpy.ndarray
            Second basis, rotates from the shared (AO) basis into the
            desired basis of the output at each k-point for each spin
            channel.

        Returns
        -------
        proj : numpy.ndarray or tuple of dyson.Lehmann
            Matrix projected into the desired basis at each k-point
            for each spin channel.
        """

        proj = lib.einsum("k...pq,sk...pi,sk...qj->sk...ij", ovlp, np.conj(mo1), mo2)

        if isinstance(matrix, np.ndarray):
            projected_matrix = lib.einsum(
                "sk...pq,sk...pi,sk...qj->sk...ij", matrix, np.conj(proj), proj
            )
        else:
            projected_matrix = [[], []]
            for s, ms in enumerate(matrix):
                for k, m in enumerate(ms):
                    coupling = lib.einsum("pk,pi->ik", m.couplings, np.conj(proj[s][k]))
                    projected_m = m.copy()
                    projected_m.couplings = coupling
                    projected_matrix[s].append(projected_m)

        return projected_matrix

    @staticmethod
    def self_energy_to_moments(se, nmom_max):
        """
        Return the hole and particle moments for a self-energy.

        Parameters
        ----------
        se : tuple of dyson.Lehmann
            Self-energy to compute the moments of at each k-point
            for each spin channel.

        Returns
        -------
        th : numpy.ndarray
            Hole moments at each k-point for each spin channel.
        tp : numpy.ndarray
            Particle moments at each k-point for each spin channel.
        """
        th = np.array([[s.occupied().moment(range(nmom_max + 1)) for s in ses] for ses in se])
        tp = np.array([[s.virtual().moment(range(nmom_max + 1)) for s in ses] for ses in se])
        return th, tp

    def build_static_potential(self, mo_energy, se):
        """
        Build the static potential approximation to the self-energy.

        Parameters
        ----------
        mo_energy : numpy.ndarray
            Molecular orbital energies at each k-point for each spin
            channel.
        se : tuple of dyson.Lehmann
            Self-energy to approximate at each k-point for each spin
            channel.

        Returns
        -------
        se_qp : numpy.ndarray
            Static potential approximation to the self-energy at each
            k-point for each spin channel.
        """
        return np.array(
            [
                [qsGW.build_static_potential(self, mo, s) for mo, s in zip(mos, ses)]
                for mos, ses in zip(mo_energy, se)
            ]
        )

    check_convergence = evKUGW.check_convergence
