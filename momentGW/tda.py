"""
Construct TDA moments.
"""

import numpy as np
import scipy.special
from pyscf import lib
from pyscf.agf2 import mpi_helper


class TDA:
    """
    Compute the self-energy moments using dTDA and numerical integration.

    Parameters
    ----------
    gw : BaseGW
        GW object.
    nmom_max : int
        Maximum moment number to calculate.
    Lpx : numpy.ndarray
        Density-fitted ERI tensor. `p` is in the basis of MOs, `x` is in
        the basis of the Green's function.
    Lia : numpy.ndarray
        Density-fitted ERI tensor for the occupied-virtual slice. `i` and
        `a` are in the basis of the screened Coulomb interaction.
    mo_energy : numpy.ndarray or tuple of numpy.ndarray, optional
        Molecular orbital energies.  If a tuple is passed, the first
        element corresponds to the Green's function basis and the second to
        the screened Coulomb interaction.  Default value is that of
        `gw._scf.mo_energy`.
    mo_occ : numpy.ndarray or tuple of numpy.ndarray, optional
        Molecular orbital occupancies.  If a tuple is passed, the first
        element corresponds to the Green's function basis and the second to
        the screened Coulomb interaction.  Default value is that of
        `gw._scf.mo_occ`.
    """

    def __init__(
        self,
        gw,
        nmom_max,
        Lpx,
        Lia,
        mo_energy=None,
        mo_occ=None,
    ):
        self.gw = gw
        self.nmom_max = nmom_max
        self.Lpx = Lpx
        self.Lia = Lia

        # Get the MO energies for G and W
        if mo_energy is None:
            self.mo_energy_g = self.mo_energy_w = gw._scf.mo_energy
        elif isinstance(mo_energy, tuple):
            self.mo_energy_g, self.mo_energy_w = mo_energy
        else:
            self.mo_energy_g = self.mo_energy_w = mo_energy

        # Get the MO occupancies for G and W
        if mo_occ is None:
            self.mo_occ_g = self.mo_occ_w = gw._scf.mo_occ
        elif isinstance(mo_occ, tuple):
            self.mo_occ_g, self.mo_occ_w = mo_occ
        else:
            self.mo_occ_g = self.mo_occ_w = mo_occ

        # Reshape ERI tensors
        self.Lia = self.Lia.reshape(self.naux, self.mpi_size(self.nov))
        self.Lpx = self.Lpx.reshape(self.naux, self.nmo, self.mpi_size(self.mo_energy_g.size))

        # Options and thresholds
        self.report_quadrature_error = True
        if "ia" in getattr(self.gw, "compression", "").split(","):
            self.compression_tol = gw.compression_tol
        else:
            self.compression_tol = None

    def kernel(self, exact=False):
        """
        Run the polarizability calculation to compute moments of the
        self-energy.
        """

        lib.logger.info(
            self.gw,
            "Constructing %s moments (nmom_max = %d)",
            self.__class__.__name__,
            self.nmom_max,
        )

        self.compress_eris()

        if exact:
            moments_dd = self.build_dd_moments_exact()
        else:
            moments_dd = self.build_dd_moments()

        moments_occ, moments_vir = self.build_se_moments(moments_dd)

        return moments_occ, moments_vir

    def compress_eris(self):
        """Compress the ERI tensors."""

        if self.compression_tol is None or self.compression_tol < 1e-14:
            return

        lib.logger.info(self.gw, "Computing compression metric for ERIs")
        cput0 = (lib.logger.process_clock(), lib.logger.perf_counter())
        naux_init = self.naux

        tmp = np.dot(self.Lia, self.Lia.T)
        tmp = mpi_helper.reduce(tmp, root=0)
        if mpi_helper.rank == 0:
            e, v = np.linalg.eigh(tmp)
            mask = np.abs(e) > self.compression_tol
            rot = v[:, mask]
        else:
            rot = np.zeros((0,))
        del tmp

        rot = mpi_helper.bcast(rot, root=0)

        self.Lia = lib.einsum("L...,LQ->Q...", self.Lia, rot)
        self.Lpx = lib.einsum("L...,LQ->Q...", self.Lpx, rot)

        lib.logger.info(
            self.gw,
            "Compressed ERI auxiliary space from %d to %d",
            naux_init,
            self.naux,
        )
        lib.logger.timer(self.gw, "compressing ERIs", *cput0)

    def build_dd_moments(self):
        """Build the moments of the density-density response."""

        cput0 = (lib.logger.process_clock(), lib.logger.perf_counter())
        lib.logger.info(self.gw, "Building density-density moments")
        lib.logger.debug(self.gw, "Memory usage: %.2f GB", self._memory_usage())

        p0, p1 = self.mpi_slice(self.nov)
        moments = np.zeros((self.nmom_max + 1, self.naux, p1 - p0))

        # Construct energy differences
        d_full = lib.direct_sum(
            "a-i->ia",
            self.mo_energy_w[self.mo_occ_w == 0],
            self.mo_energy_w[self.mo_occ_w > 0],
        ).ravel()
        d = d_full[p0:p1]

        # Get the zeroth order moment
        moments[0] = self.Lia
        cput1 = lib.logger.timer(self.gw, "zeroth moment", *cput0)

        # Get the higher order moments
        for i in range(1, self.nmom_max + 1):
            moments[i] = moments[i - 1] * d[None]
            tmp = np.dot(moments[i - 1], self.Lia.T)
            tmp = mpi_helper.allreduce(tmp)
            moments[i] += np.dot(tmp, self.Lia) * 2.0
            del tmp
            cput1 = lib.logger.timer(self.gw, "moment %d" % i, *cput1)

        return moments

    def build_dd_moments_exact(self):
        raise NotImplementedError

    def build_se_moments(self, moments_dd):
        """Build the moments of the self-energy via convolution."""

        cput0 = (lib.logger.process_clock(), lib.logger.perf_counter())
        lib.logger.info(self.gw, "Building self-energy moments")
        lib.logger.debug(self.gw, "Memory usage: %.2f GB", self._memory_usage())

        p0, p1 = self.mpi_slice(self.nov)
        q0, q1 = self.mpi_slice(self.mo_energy_g.size)

        # Setup dependent on diagonal SE
        if self.gw.diagonal_se:
            pq = p = q = "p"
            eta = np.zeros((q1 - q0, self.nmom_max + 1, self.nmo))
            fproc = lambda x: np.diag(x)
        else:
            pq, p, q = "pq", "p", "q"
            eta = np.zeros((q1 - q0, self.nmom_max + 1, self.nmo, self.nmo))
            fproc = lambda x: x

        # Get the moments in (aux|aux) and rotate to (mo|mo)
        for n in range(self.nmom_max + 1):
            eta_aux = np.dot(moments_dd[n], self.Lia.T)  # aux^2 o v
            eta_aux = mpi_helper.allreduce(eta_aux)
            for x in range(q1 - q0):
                Lp = self.Lpx[:, :, x]
                eta[x, n] = lib.einsum(f"P{p},Q{q},PQ->{pq}", Lp, Lp, eta_aux) * 2.0
        cput1 = lib.logger.timer(self.gw, "rotating DD moments", *cput0)

        # Construct the self-energy moments
        moments_occ = np.zeros((self.nmom_max + 1, self.nmo, self.nmo))
        moments_vir = np.zeros((self.nmom_max + 1, self.nmo, self.nmo))
        moms = np.arange(self.nmom_max + 1)
        for n in moms:
            fp = scipy.special.binom(n, moms)
            fh = fp * (-1) ** moms
            if np.any(self.mo_occ_g[q0:q1] > 0):
                eo = np.power.outer(self.mo_energy_g[q0:q1][self.mo_occ_g[q0:q1] > 0], n - moms)
                to = lib.einsum(f"t,kt,kt{pq}->{pq}", fh, eo, eta[self.mo_occ_g[q0:q1] > 0])
                moments_occ[n] += fproc(to)
            if np.any(self.mo_occ_g[q0:q1] == 0):
                ev = np.power.outer(self.mo_energy_g[q0:q1][self.mo_occ_g[q0:q1] == 0], n - moms)
                tv = lib.einsum(f"t,ct,ct{pq}->{pq}", fp, ev, eta[self.mo_occ_g[q0:q1] == 0])
                moments_vir[n] += fproc(tv)
        moments_occ = mpi_helper.allreduce(moments_occ)
        moments_vir = mpi_helper.allreduce(moments_vir)
        moments_occ = 0.5 * (moments_occ + moments_occ.swapaxes(1, 2))
        moments_vir = 0.5 * (moments_vir + moments_vir.swapaxes(1, 2))
        cput1 = lib.logger.timer(self.gw, "constructing SE moments", *cput1)

        return moments_occ, moments_vir

    def _memory_usage(self):
        """Return the current memory usage in GB."""
        return lib.current_memory()[0] / 1e3

    @property
    def nmo(self):
        """Number of MOs."""
        return self.gw.nmo

    @property
    def naux(self):
        """Number of auxiliaries."""
        assert self.Lpx.shape[0] == self.Lia.shape[0]
        return self.Lpx.shape[0]

    @property
    def nov(self):
        """Number of ov states in W."""
        return np.sum(self.mo_occ_w > 0) * np.sum(self.mo_occ_w == 0)

    def mpi_slice(self, n):
        """
        Return the start and end index for the current process for total
        size `n`.
        """
        return list(mpi_helper.prange(0, n, n))[0]

    def mpi_size(self, n):
        """
        Return the number of states in the current process for total size
        `n`.
        """
        p0, p1 = self.mpi_slice(n)
        return p1 - p0
