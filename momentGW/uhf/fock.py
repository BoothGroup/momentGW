"""
Fock matrix self-consistent loop for unrestricted references.
"""

import numpy as np
from dyson import Lehmann

from momentGW import mpi_helper
from momentGW.fock import FockLoop, minimize_chempot, search_chempot


class FockLoop(FockLoop):
    """
    Self-consistent loop for the density matrix via the Hartree--Fock
    self-consistent field for spin-unrestricted molecular systems.

    Parameters
    ----------
    gw : BaseUGW
        GW object.
    gf : tuple of dyson.Lehmann, optional
        Initial Green's function object for each spin channel. If
        `None`, use `gw.init_gf()`. Default value is `None`.
    se : tuple of dyson.Lehmann, optional
        Initial self-energy object for each spin channel. If passed,
        use as dynamic part of the self-energy. If `None`, self-energy
        is assumed to be static and fully defined by the Fock matrix.
        Default value is `None`.
    fock_diis_space : int, optional
        DIIS space size for the Fock matrix. Default value is `10`.
    fock_diis_min_space : int, optional
        Minimum DIIS space size for the Fock matrix. Default value is
        `1`.
    conv_tol_nelec : float, optional
        Convergence tolerance for the number of electrons. Default
        value is `1e-6`.
    conv_tol_rdm1 : float, optional
        Convergence tolerance for the density matrix. Default value is
        `1e-8`.
    max_cycle_inner : int, optional
        Maximum number of inner iterations. Default value is `100`.
    max_cycle_outer : int, optional
        Maximum number of outer iterations. Default value is `20`.
    """

    def auxiliary_shift(self, fock, se=None):
        """
        Optimise a shift in the auxiliary energies to best satisfy the
        electron number.

        Parameters
        ----------
        fock : numpy.ndarray
            Fock matrix for each spin channel.
        se : tuple of dyson.Lehmann, optional
            Self-energy for each spin channel. If `None`, use `self.se`.
            Default value is `None`.

        Returns
        -------
        se : tuple of dyson.Lehmann
            Self-energy for each spin channel.

        Notes
        -----
        If there is no dynamic part of the self-energy (`self.se` is
        `None`), this method returns `None`.
        """

        if se is None:
            se = self.se
        if se is None:
            return None

        se_α, opt_α = minimize_chempot(
            se[0],
            fock[0],
            self.nelec[0],
            x0=se[0].chempot,
            tol=self.conv_tol_nelec * 1e-2,
            maxiter=self.max_cycle_inner,
            occupancy=1,
        )

        se_β, opt_β = minimize_chempot(
            se[1],
            fock[1],
            self.nelec[1],
            x0=se[1].chempot,
            tol=self.conv_tol_nelec * 1e-2,
            maxiter=self.max_cycle_inner,
            occupancy=1,
        )

        se = (se_α, se_β)

        return se

    def search_chempot(self, gf=None):
        """Search for a chemical potential for a given Green's function.

        Parameters
        ----------
        gf : tuple of dyson.Lehmann, optional
            Green's function for each spin channel. If `None`, use
            `self.gf`. Default value is `None`.

        Returns
        -------
        chempot : tuple of float
            Chemical potential for each spin channel.
        nerr : tuple of float
            Error in the number of electrons for each spin channel.
        """

        if gf is None:
            gf = self.gf

        chempot_α, nerr_α = search_chempot(
            gf[0].energies,
            gf[0].couplings,
            self.nmo[0],
            self.nelec[0],
            occupancy=1,
        )
        chempot_β, nerr_β = search_chempot(
            gf[1].energies,
            gf[1].couplings,
            self.nmo[1],
            self.nelec[1],
            occupancy=1,
        )
        chempot = (chempot_α, chempot_β)
        nerr = abs(nerr_α) + abs(nerr_β)

        return chempot, nerr

    def solve_dyson(self, fock, se=None):
        """Solve the Dyson equation for a given Fock matrix.

        Parameters
        ----------
        fock : numpy.ndarray
            Fock matrix for each spin channel.
        se : dyson.Lehmann, optional
            Self-energy for each spin channel. If `None`, use `self.se`.
            Default value is `None`.

        Returns
        -------
        gf : tuple of dyson.Lehmann
            Green's function for each spin channel.
        nerr : float
            Error in the number of electrons.

        Notes
        -----
        If there is no dynamic part of the self-energy (`self.se` is
        `None`), this method simply diagonalises the Fock matrix and
        returns the Lehmann representation of the resulting zeroth-order
        Green's function.
        """

        if se is None:
            se = self.se

        if se is None:
            e, c = np.linalg.eigh(fock)
        else:
            e_α, c_α = se[0].diagonalise_matrix(fock[0], chempot=0.0)
            e_β, c_β = se[1].diagonalise_matrix(fock[1], chempot=0.0)
            e = (e_α, e_β)
            c = (c_α, c_β)

        e = (mpi_helper.bcast(e[0], root=0), mpi_helper.bcast(e[1], root=0))
        c = (mpi_helper.bcast(c[0], root=0), mpi_helper.bcast(c[1], root=0))

        gf = [
            Lehmann(e[0], c[0][: self.nmo[0]], chempot=se[0].chempot if se is not None else 0.0),
            Lehmann(e[1], c[1][: self.nmo[1]], chempot=se[1].chempot if se is not None else 0.0),
        ]

        chempot, nerr = self.search_chempot(gf)
        gf[0].chempot = chempot[0]
        gf[1].chempot = chempot[1]

        return tuple(gf), nerr

    def _density_error(self, rdm1, rdm1_prev):
        """Calculate the density error."""
        return max(
            np.max(np.abs(rdm1[0] - rdm1_prev[0])).real,
            np.max(np.abs(rdm1[1] - rdm1_prev[1])).real,
        )

    @property
    def naux(self):
        """Get the number of auxiliary states."""
        return (self.se[0].naux, self.se[1].naux)

    @property
    def nqmo(self):
        """Get the number of quasiparticle MOs."""
        return (self.nmo[0] + self.naux[0], self.nmo[1] + self.naux[1])

    @property
    def nelec(self):
        """Get the number of electrons."""
        return self.nocc
