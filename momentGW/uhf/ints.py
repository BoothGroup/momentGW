"""
Integral helpers with unrestricted reference.
"""

import numpy as np

from momentGW.ints import Integrals


class Integrals_α(Integrals):
    """Overload the `__name__` to signify α part"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__class__.__name__ = "Integrals (α)"


class Integrals_β(Integrals):
    """Overload the `__name__` to signify β part"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__class__.__name__ = "Integrals (β)"


class UIntegrals(Integrals):
    """
    Container for the density-fitted integrals required for GW methods.

    Parameters
    ----------
    with_df : pyscf.df.DF
        Density fitting object.
    mo_coeff : np.ndarray
        Molecular orbital coefficients for each spin channel.
    mo_occ : np.ndarray
        Molecular orbital occupations for each spin channel.
    compression : str, optional
        Compression scheme to use. Default value is `'ia'`. See
        `momentGW.gw` for more details.
    compression_tol : float, optional
        Compression tolerance. Default value is `1e-10`. See
        `momentGW.gw` for more details.
    store_full : bool, optional
        Store the full MO integrals in memory. Default value is
        `False`.
    """

    def __init__(
        self,
        with_df,
        mo_coeff,
        mo_occ,
        compression="ia",
        compression_tol=1e-10,
        store_full=False,
    ):
        self.verbose = with_df.verbose
        self.stdout = with_df.stdout

        self.with_df = with_df
        self.mo_coeff = mo_coeff
        self.mo_occ = mo_occ
        self.compression = compression
        self.compression_tol = compression_tol
        self.store_full = store_full

        self._spins = {
            0: Integrals_α(
                self.with_df,
                self.mo_coeff[0],
                self.mo_occ[0],
                compression=self.compression,
                compression_tol=self.compression_tol,
                store_full=self.store_full,
            ),
            1: Integrals_β(
                self.with_df,
                self.mo_coeff[1],
                self.mo_occ[1],
                compression=self.compression,
                compression_tol=self.compression_tol,
                store_full=self.store_full,
            ),
        }

    get_compression_metric = None

    def transform(self, do_Lpq=None, do_Lpx=True, do_Lia=True):
        """
        Transform the integrals.

        Parameters
        ----------
        do_Lpq : bool, optional
            Whether to compute the full (aux, MO, MO) array. Default
            value is `True` if `store_full` is `True`, `False`
            otherwise.
        do_Lpx : bool, optional
            Whether to compute the compressed (aux, MO, MO) array.
            Default value is `True`.
        do_Lia : bool, optional
            Whether to compute the compressed (aux, occ, vir) array.
            Default value is `True`.
        """

        self._spins[0].transform(do_Lpq=do_Lpq, do_Lpx=do_Lpx, do_Lia=do_Lia)
        self._spins[1].transform(do_Lpq=do_Lpq, do_Lpx=do_Lpx, do_Lia=do_Lia)

    def update_coeffs(self, mo_coeff_g=None, mo_coeff_w=None, mo_occ_w=None):
        """
        Update the MO coefficients for the Green's function and the
        screened Coulomb interaction.

        Parameters
        ----------
        mo_coeff_g : tuple of numpy.ndarray, optional
            Coefficients corresponding to the Green's function for each
            spin channel. Default value is `None`.
        mo_coeff_w : tuple of numpy.ndarray, optional
            Coefficients corresponding to the screened Coulomb
            interaction for each spin channel. Default value is `None`.
        mo_occ_w : tuple of numpy.ndarray, optional
            Occupations corresponding to the screened Coulomb
            interaction for each spin channel. Default value is `None`.

        Notes
        -----
        If `mo_coeff_g` is `None`, the Green's function is assumed to
        remain in the basis in which it was originally defined, and
        vice-versa for `mo_coeff_w` and `mo_occ_w`. At least one of
        `mo_coeff_g` and `mo_coeff_w` must be provided.
        """

        self._spins[0].update_coeffs(
            mo_coeff_g=mo_coeff_g[0] if mo_coeff_g is not None else None,
            mo_coeff_w=mo_coeff_w[0] if mo_coeff_w is not None else None,
            mo_occ_w=mo_occ_w[0] if mo_occ_w is not None else None,
        )
        self._spins[1].update_coeffs(
            mo_coeff_g=mo_coeff_g[1] if mo_coeff_g is not None else None,
            mo_coeff_w=mo_coeff_w[1] if mo_coeff_w is not None else None,
            mo_occ_w=mo_occ_w[1] if mo_occ_w is not None else None,
        )

    def get_j(self, dm, basis="mo"):
        """Build the J matrix.

        Parameters
        ----------
        dm : numpy.ndarray
            Density matrix for each spin channel.
        basis : str, optional
            Basis in which to build the J matrix. One of
            `("ao", "mo")`. Default value is `"mo"`.

        Returns
        -------
        vj : tuple of numpy.ndarray
            J matrix for each spin channel.
        """

        vj_αα = self._spins[0].get_j(dm[0], basis=basis, other=self._spins[0])
        vj_αβ = self._spins[0].get_j(dm[1], basis=basis, other=self._spins[1])
        vj_ββ = self._spins[1].get_j(dm[1], basis=basis, other=self._spins[1])
        vj_βα = self._spins[1].get_j(dm[0], basis=basis, other=self._spins[0])

        vj = (
            vj_αα + vj_αβ,
            vj_ββ + vj_βα,
        )

        return vj

    def get_k(self, dm, basis="mo"):
        """Build the K matrix.

        Parameters
        ----------
        dm : numpy.ndarray
            Density matrix for each spin channel.
        basis : str, optional
            Basis in which to build the K matrix. One of
            `("ao", "mo")`. Default value is `"mo"`.

        Returns
        -------
        vk : numpy.ndarray
            K matrix for each spin channel.
        """

        vk = (
            self._spins[0].get_k(dm[0], basis=basis),
            self._spins[1].get_k(dm[1], basis=basis),
        )

        return vk

    def get_fock(self, dm, h1e, **kwargs):
        """Build the Fock matrix.

        Parameters
        ----------
        dm : tuple of numpy.ndarray
            Density matrix for each spin channel.
        h1e : tuple of numpy.ndarray
            Core Hamiltonian matrix for each spin channel.
        **kwargs : dict, optional
            Additional keyword arguments for `get_jk`.

        Returns
        -------
        fock : tuple of numpy.ndarray
            Fock matrix for each spin channel.
        """

        vj, vk = self.get_jk(dm, **kwargs)
        fock = (
            h1e[0] + vj[0] - vk[0],
            h1e[1] + vj[1] - vk[1],
        )

        return fock

    def __getitem__(self, key):
        """Get the integrals for one spin."""
        return self._spins[key]

    @property
    def Lpq(self):
        """Return the (aux, MO, MO) array."""
        return (self._spins[0].Lpq, self._spins[1].Lpq)

    @property
    def Lpx(self):
        """Return the (aux, MO, MO) array."""
        return (self._spins[0].Lpx, self._spins[1].Lpx)

    @property
    def Lia(self):
        """Return the (aux, occ, vir) array."""
        return (self._spins[0].Lia, self._spins[1].Lia)

    @property
    def mo_coeff_g(self):
        """Return the MO coefficients for the Green's function."""
        return (self._spins[0].mo_coeff_g, self._spins[1].mo_coeff_g)

    @property
    def mo_coeff_w(self):
        """
        Return the MO coefficients for the screened Coulomb interaction.
        """
        return (self._spins[0].mo_coeff_w, self._spins[1].mo_coeff_w)

    @property
    def mo_occ_w(self):
        """
        Return the MO occupation numbers for the screened Coulomb interaction.
        """
        return (self._spins[0].mo_occ_w, self._spins[1].mo_occ_w)

    @property
    def nmo(self):
        """Return the number of MOs."""
        return (self._spins[0].nmo, self._spins[1].nmo)

    @property
    def nocc(self):
        """Return the number of occupied MOs."""
        return (self._spins[0].nocc, self._spins[1].nocc)

    @property
    def nvir(self):
        """Return the number of virtual MOs."""
        return (self._spins[0].nvir, self._spins[1].nvir)

    @property
    def nmo_g(self):
        """Return the number of MOs for the Green's function."""
        return (self._spins[0].nmo_g, self._spins[1].nmo_g)

    @property
    def nmo_w(self):
        """
        Return the number of MOs for the screened Coulomb interaction.
        """
        return (self._spins[0].nmo_w, self._spins[1].nmo_w)

    @property
    def nocc_w(self):
        """
        Return the number of occupied MOs for the screened Coulomb
        interaction.
        """
        return (self._spins[0].nocc_w, self._spins[1].nocc_w)

    @property
    def nvir_w(self):
        """
        Return the number of virtual MOs for the screened Coulomb
        interaction.
        """
        return (self._spins[0].nvir_w, self._spins[1].nvir_w)

    @property
    def naux(self):
        """
        Return the number of auxiliary basis functions, after the
        compression.
        """
        return (self._spins[0].naux, self._spins[1].naux)

    @property
    def naux_full(self):
        """
        Return the number of auxiliary basis functions, before the
        compression.
        """
        return self.with_df.get_naoaux()

    @property
    def is_bare(self):
        """
        Return a boolean flag indicating whether the integrals have
        no self-consistencies.
        """
        return self._mo_coeff_g is None and self._mo_coeff_w is None

    @property
    def dtype(self):
        """
        Return the dtype of the integrals.
        """
        return np.result_type(self._spins[0].dtype, self._spins[1].dtype)
