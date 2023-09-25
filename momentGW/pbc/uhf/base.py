"""
Base class for moment-constrained GW solvers with periodic boundary
conditions and unrestricted references.
"""

import numpy as np
from pyscf.lib import logger
from pyscf.pbc.mp.kump2 import get_frozen_mask, get_nmo, get_nocc

from momentGW.base import BaseGW
from momentGW.pbc.base import BaseKGW
from momentGW.uhf.base import BaseUGW


class BaseKUGW(BaseKGW, BaseUGW):  # noqa: D101
    __doc__ = BaseKGW.__doc__

    def kernel(
        self,
        nmom_max,
        mo_energy=None,
        mo_coeff=None,
        moments=None,
        integrals=None,
    ):
        """Driver for the method.

        Parameters
        ----------
        nmom_max : int
            Maximum moment number to calculate.
        mo_energy : numpy.ndarray
            Molecular orbital energies at each k-point for each spin
            channel.
        mo_coeff : numpy.ndarray
            Molecular orbital coefficients at each k-point for each spin
            channel.
        moments : tuple of numpy.ndarray, optional
            Tuple of (hole, particle) moments at each k-point for each
            spin channel, if passed then they will be used instead of
            calculating them. Default value is `None`.
        integrals : KIntegrals, optional
            Integrals object. If `None`, generate from scratch. Default
            value is `None`.
        """

        if mo_coeff is None:
            mo_coeff = self.mo_coeff
        if mo_energy is None:
            mo_energy = self.mo_energy

        cput0 = (logger.process_clock(), logger.perf_counter())
        self.dump_flags()
        logger.info(self, "nmom_max = %d", nmom_max)

        self.converged, self.gf, self.se, self._qp_energy = self._kernel(
            nmom_max,
            mo_energy,
            mo_coeff,
            integrals=integrals,
        )

        for gf, s in zip(self.gf, ["α", "β"]):
            gf_occ = gf[0].occupied().physical(weight=1e-1)
            for n in range(min(5, gf_occ.naux)):
                en = -gf_occ.energies[-(n + 1)]
                vn = gf_occ.couplings[:, -(n + 1)]
                qpwt = np.linalg.norm(vn) ** 2
                logger.note(
                    self, "IP energy level (Γ, %s) %d E = %.16g  QP weight = %0.6g", s, n, en, qpwt
                )

        for gf, s in zip(self.gf, ["α", "β"]):
            gf_vir = gf[0].virtual().physical(weight=1e-1)
            for n in range(min(5, gf_vir.naux)):
                en = gf_vir.energies[n]
                vn = gf_vir.couplings[:, n]
                qpwt = np.linalg.norm(vn) ** 2
                logger.note(
                    self, "EA energy level (Γ, %s) %d E = %.16g  QP weight = %0.6g", s, n, en, qpwt
                )

        logger.timer(self, self.name, *cput0)

        return self.converged, self.gf, self.se, self.qp_energy

    @staticmethod
    def _gf_to_occ(gf):
        return tuple(tuple(BaseGW._gf_to_occ(g, occupancy=1) for g in gs) for gs in gf)

    @staticmethod
    def _gf_to_energy(gf):
        return tuple(tuple(BaseGW._gf_to_energy(g) for g in gs) for gs in gf)

    @staticmethod
    def _gf_to_coupling(gf, mo_coeff=None):
        if mo_coeff is None:
            mo_coeff = [[None] * len(gf[0])] * 2
        return tuple(
            tuple(BaseGW._gf_to_coupling(g, mo) for g, mo in zip(gs, mos))
            for gs, mos in zip(gf, mo_coeff)
        )

    def _gf_to_mo_energy(self, gf):
        """Find the poles of a GF which best overlap with the MOs.

        Parameters
        ----------
        gf : tuple of Lehmann
            Green's function object for each k-point for each spin
            channel.

        Returns
        -------
        mo_energy : ndarray
            Updated MO energies for each k-point for each spin channel.
        """

        mo_energy = np.zeros_like(self.mo_energy)

        for s, spin in enumerate(["α", "β"]):
            for k in self.kpts.loop(1):
                check = set()
                for i in range(self.nmo[s]):
                    arg = np.argmax(gf[s][k].couplings[i] * gf[s][k].couplings[i].conj())
                    mo_energy[s][k][i] = gf[s][k].energies[arg]
                    check.add(arg)

                if len(check) != self.nmo[s]:
                    logger.warn(
                        self, f"Inconsistent quasiparticle weights for {spin} at k-point {k}!"
                    )

        return mo_energy

    get_nmo = get_nmo
    get_nocc = get_nocc
    get_frozen_mask = get_frozen_mask

    @property
    def nmo(self):
        """Return the number of molecular orbitals."""
        # PySCF returns jagged nmo with `per_kpoint=False` depending on
        # whether there is k-point dependent occupancy:
        nmo = self.get_nmo(per_kpoint=True)
        assert len(set(nmo[0])) == 1
        assert len(set(nmo[1])) == 1
        return nmo[0][0], nmo[1][0]
