import h5py
import numpy as np
from pyscf import lib
from pyscf.agf2 import mpi_helper
from scipy.special import binom

from momentGW import ints, tda


class Integrals(ints.Integrals):
    """
    Container for the tensor-hypercontracted integrals required for GW
    methods.

    Parameters
    ----------
    with_df : pyscf.df.DF
        Density fitting object.
    mo_coeff : np.ndarray
        Molecular orbital coefficients.
    mo_occ : np.ndarray
        Molecular orbital occupations.
    file_path : str, optional
        Path to the HDF5 file containing the integrals. Default value is
        `None`.
    """

    def __init__(
        self,
        with_df,
        mo_coeff,
        mo_occ,
        file_path=None,
    ):
        self.verbose = with_df.verbose
        self.stdout = with_df.stdout

        self.with_df = with_df
        self.mo_coeff = mo_coeff
        self.mo_occ = mo_occ
        self.file_path = file_path
        self.compression = None

        self._blocks = {}
        self._blocks["coll"] = None
        self._blocks["cou"] = None
        self._mo_coeff_g = None
        self._mo_coeff_w = None
        self._mo_occ_w = None

    def get_compression_metric(self):
        """Return the compression metric - not currently used in THC."""
        return None

    def import_ints(self):
        """
        Imports a HDF5 file containing a dictionary. The keys
        'collocation_matrix' and a 'coulomb_matrix' must exist, with
        shapes (MO, aux) and (aux, aux), respectively.
        """

        if self.file_path is None:
            raise ValueError("file path cannot be None for THC implementation")

        thc_eri = h5py.File(self.file_path, "r")
        coll = np.array(thc_eri["collocation_matrix"])[..., 0].T
        cou = np.array(thc_eri["coulomb_matrix"])[0, ..., 0]
        self._blocks["coll"] = coll
        self._blocks["cou"] = cou

    def transform(self, do_Lpq=True, do_Lpx=True, do_Lia=True):
        """
        Transform the integrals.

        Parameters
        ----------
        do_Lpq : bool
            If `True` contrstructs the Lp array using the mo_coeff and
            the collocation matrix. Default value is `True`. Required
            for the initial creation.
        do_Lpx : bool
            If `True` contrstructs the Lx array using the mo_coeff_g and
            the collocation matrix. Default value is `True`.
        do_Lia : bool
            If `True` contrstructs the Li and La arrays using the
            mo_coeff_w and the collocation matrix. Default value is
            `True`.
        """

        if not any([do_Lpq, do_Lpx, do_Lia]):
            return

        if self.coll is None and self.cou is None:
            self.import_ints()

        if do_Lpq:
            Lp = lib.einsum("Lp,pq->Lq", self.coll, self.mo_coeff)
            self._blocks["Lp"] = Lp

        if do_Lpx:
            Lx = lib.einsum("Lp,pq->Lq", self.coll, self.mo_coeff_g)
            self._blocks["Lx"] = Lx

        if do_Lia:
            ci = self.mo_coeff_w[:, self.mo_occ_w > 0]
            ca = self.mo_coeff_w[:, self.mo_occ_w == 0]

            Li = lib.einsum("Lp,pi->Li", self.coll, ci)
            La = lib.einsum("Lp,pa->La", self.coll, ca)

            self._blocks["Li"] = Li
            self._blocks["La"] = La

    def get_j(self, dm, basis="mo"):
        """Build the J matrix.

        Parameters
        ----------
        dm : numpy.ndarray
            Density matrix.
        basis : str, optional
            Basis in which to build the J matrix. One of
            `("ao", "mo")`. Default value is `"mo"`.

        Returns
        -------
        vj : numpy.ndarray
            J matrix.

        Notes
        -----
        The basis of `dm` must be the same as `basis`.
        """

        assert basis in ("ao", "mo")

        if basis == "ao":
            if self.coll is None and self.cou is None:
                self.import_ints()
            Lp = self.coll
            cou = self.cou
        else:
            Lp = self.Lp
            cou = self.cou

        tmp = lib.einsum("pq,Kp,Kq->K", dm, Lp, Lp)
        tmp = lib.einsum("K,KL->L", tmp, cou)
        vj = lib.einsum("L,Lr,Ls->rs", tmp, Lp, Lp)

        return vj

    def get_k(self, dm, basis="mo"):
        """Build the K matrix.

        Parameters
        ----------
        dm : numpy.ndarray
            Density matrix.
        basis : str, optional
            Basis in which to build the K matrix. One of
            `("ao", "mo")`. Default value is `"mo"`.

        Returns
        -------
        vk : numpy.ndarray
            K matrix.

        Notes
        -----
        The basis of `dm` must be the same as `basis`.
        """

        assert basis in ("ao", "mo")

        if basis == "ao":
            if self.coll is None and self.cou is None:
                self.import_ints()
            Lp = self.coll
            cou = self.cou
        else:
            Lp = self.Lp
            cou = self.cou

        tmp = lib.einsum("pq,Kp->Kq", dm, Lp)
        tmp = lib.einsum("Kq,Lq->KL", tmp, Lp)
        tmp = lib.einsum("KL,KL->KL", tmp, cou)
        tmp = lib.einsum("KL,Ks->Ls", tmp, Lp)
        vk = lib.einsum("Ls,Lr->rs", tmp, Lp)

        return vk

    @property
    def coll(self):
        """Return the (aux, MO) collocation array."""
        return self._blocks["coll"]

    @property
    def cou(self):
        """Return the (aux, aux) Coulomb array."""
        return self._blocks["cou"]

    @property
    def Lp(self):
        """Return the (aux, MO) array."""
        return self._blocks["Lp"]

    @property
    def Lx(self):
        """Return the (aux, MO) array."""
        return self._blocks["Lx"]

    @property
    def Li(self):
        """Return the (aux, W occ) array."""
        return self._blocks["Li"]

    @property
    def La(self):
        """Return the (aux, W vir) array."""
        return self._blocks["La"]

    @property
    def naux(self):
        """
        Return the number of auxiliary basis functions.
        """
        return self.cou.shape[0]

    naux_full = naux


class TDA(tda.TDA):
    """
    Compute the self-energy moments using dTDA and numerical integration
    with tensor-hypercontraction.

    Parameters
    ----------
    gw : BaseGW
        GW object.
    nmom_max : int
        Maximum moment number to calculate.
    integrals : Integrals
        Integrals object.
    mo_energy : numpy.ndarray or tuple of numpy.ndarray, optional
        Molecular orbital energies. If a tuple is passed, the first
        element corresponds to the Green's function basis and the second to
        the screened Coulomb interaction. Default value is that of
        `gw.mo_energy`.
    mo_occ : numpy.ndarray or tuple of numpy.ndarray, optional
        Molecular orbital occupancies. If a tuple is passed, the first
        element corresponds to the Green's function basis and the second to
        the screened Coulomb interaction. Default value is that of
        `gw.mo_occ`.
    """

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

    def build_dd_moments(self):
        """
        Build the moments of the density-density response using
        tensor-hypercontraction.

        Returns
        -------
        moments : numpy.ndarray
            Moments of the density-density response.

        Notes
        -----
        Unlike the standard `momentGW.tda` implementation, this method
        scales as :math:`O(N^3)` with system size instead of
        :math:`O(N^4)`.
        """

        cput0 = (lib.logger.process_clock(), lib.logger.perf_counter())
        lib.logger.info(self.gw, "Building density-density moments")
        lib.logger.debug(self.gw, "Memory usage: %.2f GB", self._memory_usage())

        zeta = np.zeros((self.nmom_max + 1, self.naux, self.naux))
        ei = self.mo_energy_w[self.mo_occ_w > 0]
        ea = self.mo_energy_w[self.mo_occ_w == 0]

        cou_occ = np.dot(self.Li, self.Li.T)
        cou_vir = np.dot(self.La, self.La.T)
        zeta[0] = cou_occ * cou_vir

        cput1 = lib.logger.timer(self.gw, "zeroth moment", *cput0)

        cou_d_left = np.zeros((self.nmom_max + 1, self.naux, self.naux))
        cou_d_only = np.zeros((self.nmom_max + 1, self.naux, self.naux))
        cou_left = np.eye(self.naux)
        cou_square = np.dot(self.cou, zeta[0])

        for i in range(1, self.nmom_max + 1):
            cou_d_left[0] = cou_left
            cou_d_left = np.roll(cou_d_left, 1, axis=0)
            cou_left = np.dot(cou_square, cou_left) * 2.0

            cou_ei_max = lib.einsum("i,Pi,Qi->PQ", ei**i, self.Li, self.Li) * pow(-1, i)
            cou_ea_max = lib.einsum("a,Pa,Qa->PQ", ea**i, self.La, self.La)
            cou_d_only[i] = cou_ea_max * cou_occ + cou_ei_max * cou_vir

            for j in range(1, i):
                cou_ei = lib.einsum("i,Pi,Qi->PQ", ei**j, self.Li, self.Li) * pow(-1, j)
                cou_ea = lib.einsum("a,Pa,Qa->PQ", ea ** (i - j), self.La, self.La) * binom(i, j)
                cou_d_only[i] += cou_ei * cou_ea
                if j == (i - 1):
                    cou_left += np.dot(self.cou, cou_d_only[j]) * 2.0
                else:
                    cou_left += (
                        np.linalg.multi_dot((self.cou, cou_d_only[i - 1 - j], cou_d_left[i - j]))
                        * 2.0
                    )

                zeta[i] += np.dot(cou_d_only[j], cou_d_left[j])

            zeta[i] += cou_d_only[i]
            zeta[i] += np.dot(zeta[0], cou_left)

            cput1 = lib.logger.timer(self.gw, "moment %d" % i, *cput1)

        return zeta

    def build_se_moments(self, zeta):
        """
        Build the moments of the self-energy via convolution with
        tensor-hypercontraction.

        Parameters
        ----------
        moments_dd : numpy.ndarray
            Moments of the density-density response.

        Returns
        -------
        moments_occ : numpy.ndarray
            Moments of the occupied self-energy.
        moments_vir : numpy.ndarray
            Moments of the virtual self-energy.
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
            zeta_prime = np.linalg.multi_dot((self.cou, zeta[n], self.cou))
            for x in range(q1 - q0):
                Lpx = lib.einsum("Pp,P->Pp", self.integrals.Lp, self.integrals.Lx[:, x])
                eta[x, n] = lib.einsum(f"P{p},Q{q},PQ->{pq}", Lpx, Lpx, zeta_prime) * 2.0
        cput1 = lib.logger.timer(self.gw, "rotating DD moments", *cput0)

        # Construct the self-energy moments
        moments_occ, moments_vir = self.convolve(eta)
        cput1 = lib.logger.timer(self.gw, "constructing SE moments", *cput1)

        return moments_occ, moments_vir

    @property
    def Li(self):
        """Return the (aux, W occ) array."""
        return self.integrals.Li

    @property
    def La(self):
        """Return the (aux, W vir) array."""
        return self.integrals.La

    @property
    def cou(self):
        """Return the (aux, aux) Coulomb array."""
        return self.integrals.cou
