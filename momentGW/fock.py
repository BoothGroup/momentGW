"""Fock matrix self-consistent loop."""

import functools
from collections import OrderedDict

import numpy as np
import scipy
from dyson import Lehmann
from pyscf import lib

from momentGW import logging, mpi_helper, util
from momentGW.logging import init_logging


class ChemicalPotentialError(ValueError):
    """Exception raised when the chemical potential cannot be found."""

    pass


def search_chempot(w, v, nphys, nelec, occupancy=2):
    """Search for a chemical potential.

    Parameters
    ----------
    w : numpy.ndarray
        Eigenvalues.
    v : numpy.ndarray
        Eigenvectors.
    nphys : int
        Number of physical states.
    nelec : int
        Number of electrons.
    occupancy : int, optional
        Number of electrons per state. Default value is `2`.

    Returns:
    -------
    chempot : float
        Chemical potential.
    error : float
        Error in the number of electrons.
    """

    if nelec == 0:
        return w[0] - 1e-6, 0.0

    nmo = v.shape[-1]
    sum0 = sum1 = 0.0

    for i in range(nmo):
        n = occupancy * np.dot(v[:nphys, i].conj().T, v[:nphys, i]).real
        sum0, sum1 = sum1, sum1 + n

        if i > 0 and sum0 <= nelec <= sum1:
            break

    if abs(sum0 - nelec) < abs(sum1 - nelec):
        homo = i - 1
        error = nelec - sum0
    else:
        homo = i
        error = nelec - sum1

    lumo = homo + 1

    if lumo == len(w):
        chempot = w[homo] + 1e-6
    else:
        chempot = 0.5 * (w[homo] + w[lumo])

    return chempot, error


def _gradient(x, se, fock, nelec, occupancy=2, buf=None):
    """Gradient of the number of electrons w.r.t shift in auxiliary
    energies.
    """

    w, v = se.diagonalise_matrix(fock, chempot=x)
    chempot, error = search_chempot(w, v, se.nphys, nelec, occupancy=occupancy)

    nocc = np.sum(w < chempot)
    nmo = se.nphys

    h1 = -np.dot(v[nmo:, nocc:].T.conj(), v[nmo:, :nocc])
    zai = -h1 / lib.direct_sum("i-a->ai", w[:nocc], w[nocc:])
    ddm = util.einsum("ai,pa,pi->", zai, v[:nmo, nocc:], v[:nmo, :nocc].conj()).real * 4
    grad = occupancy * error * ddm

    return error**2, grad


@logging.with_timer("Chemical potential optimisation")
@logging.with_status("Optimising chemical potential")
def minimize_chempot(se, fock, nelec, occupancy=2, x0=0.0, tol=1e-6, maxiter=200):
    """Optimise the shift in auxiliary energies to satisfy the electron
    number.

    Parameters
    ----------
    se : dyson.Lehmann
        Self-energy object.
    fock : numpy.ndarray
        Fock matrix.
    nelec : int
        Number of electrons.
    occupancy : int, optional
        Number of electrons per state. Default value is `2`.
    x0 : float, optional
        Initial guess value. Default value is `0.0`.
    tol : float, optional
        Threshold in the number of electrons. Default value is `1e-6`.
    maxiter : int, optional
        Maximum number of iterations. Default value is `200`.

    Returns:
    -------
    se : dyson.Lehmann
        Self-energy object.
    opt : scipy.optimize.OptimizeResult
        Result of the optimisation.
    """

    tol = tol**2  # we minimize the squared error
    dtype = np.result_type(se.dtype, fock.dtype)
    nphys = se.nphys
    naux = se.naux
    buf = np.zeros(((nphys + naux) ** 2,), dtype=dtype)
    fargs = (se, fock, nelec, occupancy, buf)

    options = dict(maxfun=maxiter, ftol=tol, xtol=tol, gtol=tol)
    kwargs = dict(x0=x0, method="TNC", jac=True, options=options)
    fun = _gradient

    opt = scipy.optimize.minimize(fun, args=fargs, **kwargs)

    new_energies = se.energies - opt.x
    se = se.copy(energies=new_energies)
    w, v = se.diagonalise_matrix(fock)
    chempot = search_chempot(w, v, se.nphys, nelec, occupancy=occupancy)[0]
    se = se.copy(chempot=chempot)

    return se, opt


class BaseFockLoop:
    """Base class for Fock loops."""

    _defaults = OrderedDict(
        fock_diis_space=10,
        fock_diis_min_space=1,
        conv_tol_nelec=1e-6,
        conv_tol_rdm1=1e-8,
        max_cycle_inner=100,
        max_cycle_outer=20,
    )

    def __init__(self, gw, gf=None, se=None, **kwargs):
        # Parameters
        self.gw = gw

        # Options
        self._opts = self._defaults.copy()
        for key, val in kwargs.items():
            if key not in self._opts:
                raise AttributeError(f"{key} is not a valid option for {self.name}")
            self._opts[key] = val

        # Attributes
        self.converged = None
        self.gf = gf if gf is not None else gw.init_gf()
        self.se = se

        # Logging
        init_logging()

    def auxiliary_shift(self, fock=None, se=None):
        """Optimise a shift in the auxiliary energies to best satisfy the
        electron number.
        """
        raise NotImplementedError

    def solve_dyson(self, fock=None, se=None, chempot=0.0):
        """Solve the Dyson equation for a given Fock matrix."""
        raise NotImplementedError

    def search_chempot(self, gf=None):
        """Search for a chemical potential."""
        raise NotImplementedError

    def _density_error(self, rdm1, rdm1_prev):
        """Calculate the density error."""
        raise NotImplementedError

    def _kernel_dynamic(self, integrals=None):
        """Driver for the Fock loop with a self-energy."""

        # Get the integrals
        if integrals is None:
            integrals = self.gw.ao2mo()

        # Initialise the DIIS object
        diis = util.DIIS()
        diis.space = self.fock_diis_space
        diis.min_space = self.fock_diis_min_space

        # Get the Green's function and the self-energy
        gf = self.gf
        se = self.se

        # Get the Fock matrix
        rdm1 = rdm1_prev = self.make_rdm1(gf=gf)
        fock = self.get_fock(integrals, rdm1)

        with logging.with_table(title="Fock loop") as table:
            table.add_column("Iter", justify="right")
            table.add_column("Cycle", justify="right")
            table.add_column("Error (nelec)", justify="right")
            table.add_column("Δ (density)", justify="right")

            converged = False
            for cycle1 in range(1, self.max_cycle_outer + 1):
                # Shift the auxiliary energies to satisfy the electron
                # number
                se = self.auxiliary_shift(fock, se=se)

                for cycle2 in range(1, self.max_cycle_inner + 1):
                    with logging.with_status(f"Iteration [{cycle1}, {cycle2}]"):
                        # Solve the Dyson equation and calculate the
                        # Fock matrix
                        gf, nerr = self.solve_dyson(fock, se=se)
                        rdm1 = self.make_rdm1(gf=gf)
                        fock = self.get_fock(integrals, rdm1)
                        fock = diis.update(fock, xerr=None)

                        # Check for convergence
                        derr = self._density_error(rdm1, rdm1_prev)
                        if (
                            cycle2 in {1, 5, 10, 50, 100, self.max_cycle_inner}
                            or derr < self.conv_tol_rdm1
                        ):
                            nerr_style = logging.rate(
                                nerr, self.conv_tol_nelec, self.conv_tol_nelec * 1e2
                            )
                            derr_style = logging.rate(
                                derr, self.conv_tol_rdm1, self.conv_tol_rdm1 * 1e2
                            )
                            table.add_row(
                                f"{cycle1}",
                                f"{cycle2}",
                                f"[{nerr_style}]{nerr:.3g}[/]",
                                f"[{derr_style}]{derr:.3g}[/]",
                            )
                        if derr < self.conv_tol_rdm1:
                            break

                        rdm1_prev = rdm1

                # Check for convergence
                if derr < self.conv_tol_rdm1 and nerr < self.conv_tol_nelec:
                    converged = True
                    break

            else:
                converged = False

            # Print the table
            logging.write(table)

        return converged, gf, se

    def _kernel_static(self, integrals=None):
        """Driver for the Fock loop without a self-energy."""

        # Get the integrals
        if integrals is None:
            integrals = self.gw.ao2mo()

        # Initialise the DIIS object
        diis = util.DIIS()
        diis.space = self.fock_diis_space
        diis.min_space = self.fock_diis_min_space

        # Get the Green's function
        gf = self.gf

        # Get the Fock matrix
        rdm1 = rdm1_prev = self.make_rdm1(gf=gf)
        fock = self.get_fock(integrals, rdm1)

        with logging.with_table(title="Fock loop") as table:
            table.add_column("Iter", justify="right")
            table.add_column("Δ (density)", justify="right")

            for cycle in range(1, self.max_cycle_inner + 1):
                with logging.with_status(f"Iteration {cycle}"):
                    # Solve the Dyson equation
                    gf = self.solve_dyson(fock)
                    gf.chempot, _ = self.search_chempot(gf=gf)

                    # Calculate the Fock matrix
                    rdm1 = self.make_rdm1(gf=gf)
                    fock = self.get_fock(integrals, rdm1)
                    fock = diis.update(fock, xerr=None)

                    # Check for convergence
                    derr = np.max(np.absolute(rdm1 - rdm1_prev))
                    if (
                        cycle in {1, 5, 10, 50, 100, self.max_cycle_inner}
                        or derr < self.conv_tol_rdm1
                    ):
                        style = logging.rate(derr, self.conv_tol_rdm1, self.conv_tol_rdm1 * 1e2)
                        table.add_row(f"{cycle}", f"[{style}]{derr:.3g}[/]")
                    if derr < self.conv_tol_rdm1:
                        converged = True
                        break

                    rdm1_prev = rdm1.copy()

            else:
                converged = False

            # Print the table
            logging.write(table)

        return converged, gf, None

    @functools.cached_property
    def h1e(self):
        """Get the core Hamiltonian."""
        with util.SilentSCF(self.gw._scf):
            h1e = util.einsum(
                "...pq,...pi,...qj->...ij",
                self.gw._scf.get_hcore(),
                np.conj(self.mo_coeff),
                self.mo_coeff,
            )
        return h1e

    def make_rdm1(self, gf=None):
        """Get the first-order reduced density matrix.

        Parameters
        ----------
        gf : dyson.Lehmann, optional
            Green's function object. If `None`, use either `self.gf`, or
            the mean-field Green's function. Default value is `None`.

        Returns:
        -------
        rdm1 : numpy.ndarray
            First-order reduced density matrix.
        """
        if gf is None:
            gf = self.gf
        return self.gw.make_rdm1(gf=gf)

    def get_fock(self, integrals, rdm1, h1e=None):
        """Get the Fock matrix.

        Parameters
        ----------
        integrals : BaseIntegrals
            Integrals object.
        rdm1 : numpy.ndarray
            First-order reduced density matrix.
        h1e : numpy.ndarray, optional
            Core Hamiltonian. If `None`, use `self.h1e`. Default value
            is `None`.

        Returns:
        -------
        fock : numpy.ndarray
            Fock matrix.
        """
        if h1e is None:
            h1e = self.h1e
        return integrals.get_fock(rdm1, h1e)

    @property
    def mo_coeff(self):
        """Get the MO coefficients."""
        return self.gw.mo_coeff

    @property
    def nmo(self):
        """Get the number of MOs."""
        return self.gw.nmo

    @property
    def nocc(self):
        """Get the number of occupied MOs."""
        return self.gw.nocc

    def __getattr__(self, key):
        """Try to get an attribute from the `_opts` dictionary. If it is
        not found, raise an `AttributeError`.

        Parameters
        ----------
        key : str
            Attribute key.

        Returns:
        -------
        value : any
            Attribute value.
        """
        if key in self._defaults:
            return self._opts[key]
        return self.__getattribute__(key)

    def __setattr__(self, key, val):
        """Try to set an attribute from the `_opts` dictionary. If it is
        not found, raise an `AttributeError`.

        Parameters
        ----------
        key : str
            Attribute key.
        """
        if key in self._defaults:
            self._opts[key] = val
        else:
            super().__setattr__(key, val)


class FockLoop(BaseFockLoop):
    """Self-consistent loop for the density matrix via the Hartree--Fock
    self-consistent field for spin-restricted molecular systems.

    Parameters
    ----------
    gw : BaseGW
        GW object.
    gf : dyson.Lehmann, optional
        Initial Green's function object. If `None`, use `gw.init_gf()`.
        Default value is `None`.
    se : dyson.Lehmann, optional
        Initial self-energy object. If passed, use as dynamic part of
        the self-energy. If `None`, self-energy is assumed to be static
        and fully defined by the Fock matrix. Default value is `None`.
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
        """Optimise a shift in the auxiliary energies to best satisfy the
        electron number.

        Parameters
        ----------
        fock : numpy.ndarray
            Fock matrix.
        se : dyson.Lehmann, optional
            Self-energy. If `None`, use `self.se`. Default value is
            `None`.

        Returns:
        -------
        se : dyson.Lehmann
            Self-energy.

        Notes:
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
        se, opt = minimize_chempot(
            se,
            fock,
            self.nelec,
            x0=se.chempot,
            tol=self.conv_tol_nelec,
            maxiter=self.max_cycle_inner,
        )

        return se

    def search_chempot(self, gf=None):
        """Search for a chemical potential for a given Green's function.

        Parameters
        ----------
        gf : dyson.Lehmann, optional
            Green's function. If `None`, use `self.gf`. Default value is
            `None`.

        Returns:
        -------
        chempot : float
            Chemical potential.
        nerr : float
            Error in the number of electrons.
        """

        # Get the Green's function
        if gf is None:
            gf = self.gf

        # Search for the chemical potential
        chempot, nerr = search_chempot(gf.energies, gf.couplings, self.nmo, self.nelec)
        nerr = abs(nerr)

        return chempot, nerr

    def solve_dyson(self, fock, se=None):
        """Solve the Dyson equation for a given Fock matrix.

        Parameters
        ----------
        fock : numpy.ndarray
            Fock matrix.
        se : dyson.Lehmann, optional
            Self-energy. If `None`, use `self.se`. Default value is
            `None`.

        Returns:
        -------
        gf : dyson.Lehmann
            Green's function.
        nerr : float
            Error in the number of electrons.

        Notes:
        -----
        If there is no dynamic part of the self-energy (`self.se` is
        `None`), this method simply diagonalises the Fock matrix and
        returns the Lehmann representation of the resulting zeroth-order
        Green's function.
        """

        # Get the self-energy
        if se is None:
            se = self.se

        # Diagonalise the (extended) Fock matrix
        if se is None:
            e, c = np.linalg.eigh(fock)
        else:
            e, c = se.diagonalise_matrix(fock, chempot=0.0)

        # Broadcast the eigenvalues and eigenvectors in case of
        # hybrid parallelisation introducing non-determinism
        e = mpi_helper.bcast(e, root=0)
        c = mpi_helper.bcast(c, root=0)

        # Construct the Green's function
        gf = Lehmann(e, c[: self.nmo], chempot=se.chempot if se is not None else 0.0)

        # Search for the chemical potential
        chempot, nerr = self.search_chempot(gf)
        gf = gf.copy(chempot=chempot)

        return gf, nerr

    @logging.with_timer("Fock loop")
    @logging.with_status("Running Fock loop")
    def kernel(self, integrals=None):
        """Driver for the Fock loop.

        Parameters
        ----------
        integrals : Integrals, optional
            Integrals object. If `None`, generate from scratch. Default
            value is `None`.

        Returns:
        -------
        converged : bool
            Whether the loop has converged.
        gf : dyson.Lehmann
            Green's function object.
        se : dyson.Lehmann
            Self-energy object.
        """

        # Get the kernel
        if self.se is None:
            kernel = self._kernel_static
        else:
            kernel = self._kernel_dynamic

        # Run the kernel
        self.converged, self.gf, self.se = kernel(integrals=integrals)

        return self.converged, self.gf, self.se

    def _density_error(self, rdm1, rdm1_prev):
        """Calculate the density error.

        Parameters
        ----------
        rdm1 : numpy.ndarray
            Current density matrix.
        rdm1_prev : numpy.ndarray
            Previous density matrix.

        Returns:
        -------
        error : float
            Density error.
        """
        return np.max(np.abs(rdm1 - rdm1_prev)).real

    @property
    def naux(self):
        """Get the number of auxiliary states."""
        return self.se.naux

    @property
    def nqmo(self):
        """Get the number of quasiparticle MOs."""
        return self.nmo + self.naux

    @property
    def nelec(self):
        """Get the number of electrons."""
        return self.nocc * 2
