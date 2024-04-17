"""
Tensor hyper-contraction.
"""

import h5py
import numpy as np
from scipy.special import binom

from momentGW import init_logging, ints, logging, util
from momentGW.tda import dTDA as DFdTDA


class Integrals(ints.Integrals):
    """
    Container for the tensor-hypercontracted integrals required for GW
    methods.

    Parameters
    ----------
    with_df : pyscf.df.DF
        Density fitting object.
    mo_coeff : numpy.ndarray
        Molecular orbital coefficients.
    mo_occ : numpy.ndarray
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
        # Parameters
        self.with_df = with_df
        self.mo_coeff = mo_coeff
        self.mo_occ = mo_occ
        self.file_path = file_path

        # Options
        self.compression = None

        # Logging
        init_logging()

        # Attributes
        self._blocks = {}
        self._blocks["coll"] = None
        self._blocks["cou"] = None
        self._mo_coeff_g = None
        self._mo_coeff_w = None
        self._mo_occ_w = None
        self._rot = None

    def get_compression_metric(self):
        """Return the compression metric - not currently used in THC."""
        return None

    def import_thc_components(self):
        """
        Import a HDF5 file containing a dictionary. The keys
        `"collocation_matrix"` and a `"coulomb_matrix"` must exist, with
        shapes ``(MO, aux)`` and ``(aux, aux)``, respectively.
        """

        if self.file_path is None:
            raise ValueError("file path cannot be None for THC implementation")

        thc_eri = h5py.File(self.file_path, "r")
        coll = np.array(thc_eri["collocation_matrix"])[..., 0].T
        cou = np.array(thc_eri["coulomb_matrix"])[0, ..., 0]
        self._blocks["coll"] = coll
        self._blocks["cou"] = cou

        self._naux = self.cou.shape[0]

    @logging.with_status("Transforming integrals")
    def transform(self, do_Lpq=True, do_Lpx=True, do_Lia=True):
        """
        Transform the integrals in-place.

        Parameters
        ----------
        do_Lpq : bool, optional
            Whether the ``(aux, MO, MO)`` array is required. In THC,
            this requires the `Lp` array. Default value is `True`.
        do_Lpx : bool, optional
            Whether the ``(aux, MO, MO)`` array is required. In THC,
            this requires the `Lx` array. Default value is `True`.
        do_Lia : bool, optional
            Whether the ``(aux, occ, vir)`` array is required. In THC,
            this requires the `Li` and `La` arrays. Default value is
            `True`.
        """

        # Check if any arrays are required
        if not any([do_Lpq, do_Lpx, do_Lia]):
            return

        # Import THC components
        if self.coll is None and self.cou is None:
            self.import_thc_components()

        # Transform the (L|pq) array
        if do_Lpq:
            Lp = util.einsum("Lp,pq->Lq", self.coll, self.mo_coeff)
            self._blocks["Lp"] = Lp

        # Transform the (L|px) array
        if do_Lpx:
            Lx = util.einsum("Lp,pq->Lq", self.coll, self.mo_coeff_g)
            self._blocks["Lx"] = Lx

        # Transform the (L|ia) and (L|ai) arrays
        if do_Lia:
            ci = self.mo_coeff_w[:, self.mo_occ_w > 0]
            ca = self.mo_coeff_w[:, self.mo_occ_w == 0]

            Li = util.einsum("Lp,pi->Li", self.coll, ci)
            La = util.einsum("Lp,pa->La", self.coll, ca)

            self._blocks["Li"] = Li
            self._blocks["La"] = La

    @logging.with_timer("J matrix")
    @logging.with_status("Building J matrix")
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

        # Check the input
        assert basis in ("ao", "mo")

        # Get the components
        if basis == "ao":
            if self.coll is None and self.cou is None:
                self.import_thc_components()
            Lp = self.coll
            cou = self.cou
        else:
            Lp = self.Lp
            cou = self.cou

        # Build the J matrix
        tmp = util.einsum("pq,Kp,Kq->K", dm, Lp, Lp)
        tmp = util.einsum("K,KL->L", tmp, cou)
        vj = util.einsum("L,Lr,Ls->rs", tmp, Lp, Lp)

        return vj

    @logging.with_timer("K matrix")
    @logging.with_status("Building K matrix")
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

        # Check the input
        assert basis in ("ao", "mo")

        # Get the components
        if basis == "ao":
            if self.coll is None and self.cou is None:
                self.import_thc_components()
            Lp = self.coll
            cou = self.cou
        else:
            Lp = self.Lp
            cou = self.cou

        # Build the K matrix
        tmp = util.einsum("pq,Kp->Kq", dm, Lp)
        tmp = util.einsum("Kq,Lq->KL", tmp, Lp)
        tmp = util.einsum("KL,KL->KL", tmp, cou)
        tmp = util.einsum("KL,Ks->Ls", tmp, Lp)
        vk = util.einsum("Ls,Lr->rs", tmp, Lp)

        return vk

    @property
    def coll(self):
        """Get the ``(aux, MO)`` collocation array."""
        return self._blocks["coll"]

    @property
    def cou(self):
        """Get the ``(aux, aux)`` Coulomb array."""
        return self._blocks["cou"]

    @property
    def Lp(self):
        """Get the ``(aux, MO)`` array."""
        return self._blocks["Lp"]

    @property
    def Lx(self):
        """Get the ``(aux, MO)`` array."""
        return self._blocks["Lx"]

    @property
    def Li(self):
        """Get the ``(aux, W occ)`` array."""
        return self._blocks["Li"]

    @property
    def La(self):
        """Get the ``(aux, W vir)`` array."""
        return self._blocks["La"]


class dTDA(DFdTDA):
    """
    Compute the self-energy moments using dTDA and numerical integration
    with tensor-hypercontraction.

    Parameters
    ----------
    gw : BaseGW
        GW object.
    nmom_max : int
        Maximum moment number to calculate.
    integrals : BaseIntegrals
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

    @logging.with_timer("Density-density moments")
    @logging.with_status("Constructing density-density moments")
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

        # Initialise the moments
        zeta = np.zeros((self.nmom_max + 1, self.naux, self.naux))
        ei = self.mo_energy_w[self.mo_occ_w > 0]
        ea = self.mo_energy_w[self.mo_occ_w == 0]

        # Get the zeroth order moment
        cou_occ = np.dot(self.Li, self.Li.T)
        cou_vir = np.dot(self.La, self.La.T)
        zeta[0] = cou_occ * cou_vir

        # Initialise intermediate arrays
        cou_d_left = np.zeros((self.nmom_max + 1, self.naux, self.naux))
        cou_d_only = np.zeros((self.nmom_max + 1, self.naux, self.naux))
        cou_left = np.eye(self.naux)
        cou_square = np.dot(self.cou, zeta[0])

        for i in range(1, self.nmom_max + 1):
            # Update intermediate arrays
            cou_d_left[0] = cou_left
            cou_d_left = np.roll(cou_d_left, 1, axis=0)
            cou_left = np.dot(cou_square, cou_left) * 2.0

            cou_ei_max = util.einsum("i,Pi,Qi->PQ", ei**i, self.Li, self.Li) * pow(-1, i)
            cou_ea_max = util.einsum("a,Pa,Qa->PQ", ea**i, self.La, self.La)
            cou_d_only[i] = cou_ea_max * cou_occ + cou_ei_max * cou_vir

            for j in range(1, i):
                cou_ei = util.einsum("i,Pi,Qi->PQ", ei**j, self.Li, self.Li) * pow(-1, j)
                cou_ea = util.einsum("a,Pa,Qa->PQ", ea ** (i - j), self.La, self.La) * binom(i, j)
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

        return zeta

    @logging.with_timer("Self-energy moments")
    @logging.with_status("Constructing self-energy moments")
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
                Lpx = util.einsum("Pp,P->Pp", self.integrals.Lp, self.integrals.Lx[:, x + q0])
                eta[x, n] = util.einsum(f"P{p},Q{q},PQ->{pq}", Lpx, Lpx, zeta_prime) * 2.0

        # Construct the self-energy moments
        moments_occ, moments_vir = self.convolve(eta)

        return moments_occ, moments_vir

    @property
    def Li(self):
        """Get the ``(aux, W occ)`` array."""
        return self.integrals.Li

    @property
    def La(self):
        """Get the ``(aux, W vir)`` array."""
        return self.integrals.La

    @property
    def cou(self):
        """Get the ``(aux, aux)`` Coulomb array."""
        return self.integrals.cou
