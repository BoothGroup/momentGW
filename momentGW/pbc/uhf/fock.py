"""Fock matrix and static self-energy parts with periodic boundary conditions and unrestricted
references."""

import numpy as np
from dyson import Lehmann

from momentGW import logging, mpi_helper
from momentGW.pbc.fock import FockLoop, minimize_chempot, search_chempot


class FockLoop(FockLoop):
    """Self-consistent loop for the density matrix via the Hartree--Fock self-consistent field for
    spin-unrestricted periodic systems.

    Parameters
    ----------
    gw : BaseKUGW
        GW object.
    gf : tuple of tuple of dyson.Lehmann, optional
        Initial Green's function object at each k-point for each spin
        channel. If `None`, use `gw.init_gf()`. Default value is `None`.
    se : tuple of tuple of dyson.Lehmann, optional
        Initial self-energy object at each k-point for each spin
        channel. If passed, use as dynamic part of the self-energy. If
        `None`, self-energy is assumed to be static and fully defined by
        the Fock matrix. Default value is `None`.
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
        """Optimise a shift in the auxiliary energies to best satisfy the electron number.

        Parameters
        ----------
        fock : numpy.ndarray
            Fock matrix at each k-point for each spin channel.
        se : tuple of tuple of dyson.Lehmann, optional
            Self-energy at each k-point for each spin channel. If
            `None`, use `self.se`. Default value is `None`.

        Returns
        -------
        se : tuple of tuple of dyson.Lehmann
            Self-energy at each k-point for each spin channel.

        Notes
        -----
        If there is no dynamic part of the self-energy (`self.se` is
        `None`), this method returns `None`.
        """

        # Get the self-energy
        if se is None:
            se = self.se
        if se is None:
            return None

        # Optimise the shift in the auxiliary energies
        se_α, opt_α = minimize_chempot(
            se[0],
            fock[0],
            sum(self.nelec[0]),
            x0=se[0][0].chempot,
            tol=self.conv_tol_nelec,
            maxiter=self.max_cycle_inner,
            occupancy=1,
        )
        se_β, opt_β = minimize_chempot(
            se[1],
            fock[1],
            sum(self.nelec[1]),
            x0=se[1][0].chempot,
            tol=self.conv_tol_nelec,
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

        # Get the Green's function
        if gf is None:
            gf = self.gf

        # Search for the chemical potential
        chempot_α, nerr_α = search_chempot(
            [g.energies for g in gf[0]],
            [g.couplings for g in gf[0]],
            self.nmo[0],
            sum(self.nelec[0]),
            occupancy=1,
        )
        chempot_β, nerr_β = search_chempot(
            [g.energies for g in gf[1]],
            [g.couplings for g in gf[1]],
            self.nmo[1],
            sum(self.nelec[1]),
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
            Fock matrix at each k-point for each spin channel.
        se : tuple of dyson.Lehmann, optional
            Self-energy at each k-point. If `None`, use `self.se`.
            Default value is `None`.

        Returns
        -------
        gf : tuple of dyson.Lehmann
            Green's function at each k-point.
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

        # Diagonalise the (extended) Fock matrix
        if se is None:
            e, c = np.linalg.eigh(fock)
        else:
            e_α, c_α = zip(*[s.diagonalise_matrix(f, chempot=0.0) for s, f in zip(se[0], fock[0])])
            e_β, c_β = zip(*[s.diagonalise_matrix(f, chempot=0.0) for s, f in zip(se[1], fock[1])])
            e = (e_α, e_β)
            c = (c_α, c_β)

        # Broadcast the eigenvalues and eigenvectors in case of
        # hybrid parallelisation introducing non-determinism
        e = [
            [mpi_helper.bcast(ek, root=0) for ek in e[0]],
            [mpi_helper.bcast(ek, root=0) for ek in e[1]],
        ]
        c = [
            [mpi_helper.bcast(ck, root=0) for ck in c[0]],
            [mpi_helper.bcast(ck, root=0) for ck in c[1]],
        ]

        # Construct the Green's function
        gf = [
            [Lehmann(ek, ck[: self.nmo[0]], chempot=0.0) for ek, ck in zip(e[0], c[0])],
            [Lehmann(ek, ck[: self.nmo[1]], chempot=0.0) for ek, ck in zip(e[1], c[1])],
        ]

        # Search for the chemical potential
        chempot, nerr = self.search_chempot(gf)
        for k in self.kpts.loop(1):
            gf[0][k].chempot = chempot[0]
            gf[1][k].chempot = chempot[1]

        return tuple(tuple(gf_s) for gf_s in gf), nerr

    @logging.with_timer("Fock loop")
    @logging.with_status("Running Fock loop")
    def kernel(self, integrals=None):
        """Driver for the Fock loop.

        Parameters
        ----------
        integrals : KUIntegrals, optional
            Integrals object. If `None`, generate from scratch. Default
            value is `None`.

        Returns
        -------
        converged : bool
            Whether the loop has converged.
        gf : tuple of tuple of dyson.Lehmann
            Green's function object at each k-point for each spin
            channel.
        se : tuple of tuple of dyson.Lehmann
            Self-energy object at each k-point for each spin channel.
        """
        return super().kernel(integrals)

    def _density_error(self, rdm1, rdm1_prev):
        """Calculate the density error."""
        return max(
            np.max(np.abs(rdm1[0] - rdm1_prev[0])).real,
            np.max(np.abs(rdm1[1] - rdm1_prev[1])).real,
        )

    @property
    def naux(self):
        """Get the number of auxiliary states."""
        return (tuple(s.naux for s in self.se[0]), tuple(s.naux for s in self.se[1]))

    @property
    def nqmo(self):
        """Get the number of quasiparticle MOs."""
        return (
            tuple(s.nphys + s.naux for s in self.se[0]),
            tuple(s.nphys + s.naux for s in self.se[1]),
        )

    @property
    def nelec(self):
        """Get the number of electrons."""
        return self.nocc
