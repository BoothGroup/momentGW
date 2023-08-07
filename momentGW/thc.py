import numpy as np
from h5py import File
from pyscf import lib
from pyscf.agf2 import mpi_helper
from scipy.special import binom

from momentGW import tda


class Integrals:
    """
    Container for the integrals required for GW methods.
    """

    def __init__(
        self,
        with_df,
        mo_coeff,
        mo_occ,
        file_path=None,
    ):
        self.with_df = with_df
        self.mo_coeff = mo_coeff
        self.mo_occ = mo_occ
        self.file_path = file_path

        self._blocks = {}

    def transform(self):
        """
        Imports a H5PY file containing a dictionary. Inside the dict, a
        'collocation_matrix' and a 'coulomb_matrix' must be contained
        with shapes (aux, MO) and (aux,aux) respectively.
        """

        if self.file_path is None:
            raise ValueError("file path cannot be None for THC implementation")

        thc_eri = File(self.file_path, "r")
        coll = np.array(thc_eri["collocation_matrix"]).T[0].T
        cou = np.array(thc_eri["coulomb_matrix"][0]).T[0].T
        Xip = coll[: self.nocc, :]
        Xap = coll[self.nocc :, :]

        self._blocks["coll"] = coll
        self._blocks["cou"] = cou
        self._blocks["Xip"] = Xip
        self._blocks["Xap"] = Xap

    @property
    def Coll(self):
        """
        Return the (aux, MO) array.
        """
        return self._blocks["coll"]

    @property
    def Cou(self):
        """
        Return the (aux, aux) array.
        """
        return self._blocks["cou"]

    @property
    def Xip(self):
        """
        Return the (aux, W occ) array.
        """
        return self._blocks["Xip"]

    @property
    def Xap(self):
        """
        Return the (aux, W vir) array.
        """
        return self._blocks["Xap"]

    @property
    def nmo(self):
        """
        Return the number of MOs.
        """
        return self.mo_coeff.shape[-1]

    @property
    def nocc(self):
        """
        Return the number of occupied MOs.
        """
        return np.sum(self.mo_occ > 0)

    @property
    def naux(self):
        """
        Return the number of auxiliary basis functions, after the
        compression.
        """
        return self.Cou.shape[0]


class TDA(tda.TDA):
    def __init__(
        self,
        gw,
        nmom_max,
        integrals,
        mo_energy=None,
        mo_occ=None,
    ):
        self.gw = gw
        self.integrals = integrals
        self.nmom_max = nmom_max

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

        # Options and thresholds
        self.report_quadrature_error = True
        if "ia" in getattr(self.gw, "compression", "").split(","):
            self.compression_tol = gw.compression_tol
        else:
            self.compression_tol = None

    def build_Z_prime(self):
        """
        Form the X_iP X_aP X_iQ X_aQ = Z_X contraction at N^3 cost.
        """

        Y_i_PQ = np.einsum("iP,iQ->PQ", self.XiP, self.XiP)
        Y_a_PQ = np.einsum("aP,aQ->PQ", self.XaP, self.XaP)
        Z_X_PQ = np.einsum("PQ,PQ->PQ", Y_i_PQ, Y_a_PQ)

        return Z_X_PQ

    def build_dd_moments(self):
        """
        Calcualte the moments recusively, in a form similiar to that of
        a density-density response, at N^3 cost using only THC elements.
        """

        cput0 = (lib.logger.process_clock(), lib.logger.perf_counter())
        lib.logger.info(self.gw, "Building density-density moments")
        lib.logger.debug(self.gw, "Memory usage: %.2f GB", self._memory_usage())

        zeta = np.zeros((self.nmom_max + 1, self.XiP.shape[1], self.XiP.shape[1]))
        ZD_left = np.zeros((self.nmom_max + 1, self.naux, self.naux))
        ZD_only = np.zeros((self.nmom_max + 1, self.naux, self.naux))
        ei = self.mo_energy_w[self.mo_occ_w > 0]
        ea = self.mo_energy_w[self.mo_occ_w == 0]

        Z_prime = self.build_Z_prime()
        ZZ = np.einsum("PQ,QR->PR", self.Z, Z_prime)

        zeta[0] = Z_prime

        cput1 = lib.logger.timer(self.gw, "Zeta zero", *cput0)

        YaP = np.einsum("aP,aQ->PQ", self.XaP, self.XaP)
        YiP = np.einsum("aP,aQ->PQ", self.XiP, self.XiP)

        Z_left = np.eye((self.naux))

        for i in range(1, self.nmom_max + 1):
            ZD_left[0] = Z_left
            ZD_left = np.roll(ZD_left, 1, axis=0)

            Z_left = np.einsum("PQ,QR->PR", ZZ, Z_left) * 2

            Yei_max = np.einsum("i,iP,iQ->PQ", (-1) ** (i) * ei ** (i), self.XiP, self.XiP)
            Yea_max = np.einsum("a,aP,aQ->PQ", ea ** (i), self.XaP, self.XaP)
            ZD_only[i] = np.einsum("PQ,PQ->PQ", Yea_max, YiP) + np.einsum("PQ,PQ->PQ", Yei_max, YaP)
            ZD_temp = np.zeros((self.naux, self.naux))
            for j in range(1, i):
                Yei = np.einsum("i,iP,iQ->PQ", (-1) ** (j) * ei ** (j), self.XiP, self.XiP)
                Yea = np.einsum("a,aP,aQ->PQ", binom(i, j) * ea ** (i - j), self.XaP, self.XaP)
                ZD_only[i] += np.einsum("PQ,PQ->PQ", Yea, Yei)
                if j == i - 1:
                    Z_left += np.einsum("PQ,QR->PR", self.Z, ZD_only[j]) * 2
                else:
                    Z_left += (
                        np.einsum("PQ,QR,RS->PS", self.Z, ZD_only[i - 1 - j], ZD_left[i - j]) * 2
                    )
                ZD_temp += np.einsum("PQ,QR->PR", ZD_only[j], ZD_left[j])
            zeta[i] = ZD_only[i] + ZD_temp + np.einsum("PQ,QR->PR", Z_prime, Z_left)
            cput1 = lib.logger.timer(self.gw, "Zeta %d" % i, *cput1)

        return zeta

    def build_se_moments(self, zeta):
        """
        Build the moments of the self-energy via convolution.
        """

        cput0 = (lib.logger.process_clock(), lib.logger.perf_counter())
        lib.logger.info(self.gw, "Building self-energy moments")
        lib.logger.debug(self.gw, "Memory usage: %.2f GB", self._memory_usage())

        # Setup dependent on diagonal SE
        q0, q1 = self.mpi_slice(self.mo_energy_g.size)
        if self.gw.diagonal_se:
            eta = np.zeros((q1 - q0, self.nmom_max + 1, self.nmo))
            pq = p = q = "p"
        else:
            eta = np.zeros((q1 - q0, self.nmom_max + 1, self.nmo, self.nmo))
            pq, p, q = "pq", "p", "q"

        # Get the moments in (aux|aux) and rotate to (mo|mo)
        for n in range(self.nmom_max + 1):
            zeta_prime = np.linalg.multi_dot((self.Z, zeta[n], self.Z))
            for x in range(q1 - q0):
                Lp = lib.einsum("pP,P->Pp", self.integrals.Coll, self.integrals.Coll[x])
                eta[x, n] = np.einsum(f"P{p},Q{q},PQ->{pq}", Lp, Lp, zeta_prime) * 2.0
        cput1 = lib.logger.timer(self.gw, "rotating DD moments", *cput0)

        # Construct the self-energy moments
        moments_occ, moments_vir = self.convolve(eta)
        cput1 = lib.logger.timer(self.gw, "constructing SE moments", *cput1)

        return moments_occ, moments_vir

    @property
    def XiP(self):
        return self.integrals.Xip

    @property
    def XaP(self):
        return self.integrals.XaP

    @property
    def Z(self):
        return self.integrals.Cou
