"""
Base class for moment-constained GW solvers with unrestricted
references.
"""

from pyscf.mp.ump2 import get_frozen_mask, get_nmo, get_nocc

from momentGW.base import BaseGW


class BaseUGW(BaseGW):
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
            Molecular orbital energies for each spin channel.
        mo_coeff : numpy.ndarray
            Molecular orbital coefficients for each spin channel.
        moments : tuple of numpy.ndarray, optional
            Tuple of (hole, particle) moments for each spin channel, if
            passed then they will be used instead of calculating them.
            Default value is `None`.
        integrals : UIntegrals, optional
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

        for gf in self.gf:
            gf_occ = self.gf.get_occupied()
            gf_occ.remove_uncoupled(tol=1e-1)
            for n in range(min(5, gf_occ.naux)):
                en = -gf_occ.energy[-(n + 1)]
                vn = gf_occ.coupling[:, -(n + 1)]
                qpwt = np.linalg.norm(vn) ** 2
                logger.note(self, "IP energy level (α) %d E = %.16g  QP weight = %0.6g", n, en, qpwt)

        for gf in self.gf:
            gf_vir = self.gf.get_virtual()
            gf_vir.remove_uncoupled(tol=1e-1)
            for n in range(min(5, gf_vir.naux)):
                en = gf_vir.energy[n]
                vn = gf_vir.coupling[:, n]
                qpwt = np.linalg.norm(vn) ** 2
                logger.note(self, "EA energy level (β) %d E = %.16g  QP weight = %0.6g", n, en, qpwt)

        logger.timer(self, self.name, *cput0)

        return self.converged, self.gf, self.se, self.qp_energy

    @property
    def qp_energy(self):
        """
        Return the quasiparticle energies for each spin channel. For
        most GW methods, this simply consists of the poles of the
        `self.gf` that best overlap with the MOs, in order. In some
        methods such as qsGW, these two quantities are not the same.
        """

        if self._qp_energy is not None:
            return self._qp_energy

        qp_energy = (
            self._gf_to_mo_energy(self.gf[0]),
            self._gf_to_mo_energy(self.gf[1]),
        )

        return qp_energy

    get_nmo = get_nmo
    get_nocc = get_nocc
    get_frozen_mask = get_frozen_mask
