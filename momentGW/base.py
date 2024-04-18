"""
Base classes for moment-constrained GW solvers.
"""

import functools
from collections import OrderedDict

import numpy as np

from momentGW import logging, mpi_helper, util
from momentGW.logging import init_logging


class Base:
    """Base class."""

    # Default options
    _defaults = OrderedDict()

    def _convert_mf(self, mf):
        """Abstract method for converting the mean-field object."""
        raise NotImplementedError

    def __init__(
        self,
        mf,
        mo_energy=None,
        mo_coeff=None,
        mo_occ=None,
        frozen=None,
        **kwargs,
    ):
        # Options
        self._opts = self._defaults.copy()
        for key, val in kwargs.items():
            if key not in self._opts:
                raise AttributeError(f"{key} is not a valid option for {self.name}")
            self._opts[key] = val

        # Parameters
        self._scf = self._convert_mf(mf)
        self._mo_energy = mo_energy
        self._mo_coeff = mo_coeff
        self._mo_occ = mo_occ
        self._frozen = frozen

        # Logging
        init_logging()

    def _opt_is_used(self, key):
        """
        Check if an option is used by the solver. This is useful for
        determining whether to print the option in the table.

        Parameters
        ----------
        key : str
            Option key.

        Returns
        -------
        used : bool
            Whether the option is used.
        """
        if key == "fock_opts":
            return self.fock_loop
        if key == "thc_opts":
            return self.polarizability.lower().startswith("thc")
        if key == "npoints":
            return self.polarizability.lower().endswith("drpa")
        if key == "eta":
            return self.srg == 0.0
        if key == "srg":
            return self.srg != 0.0
        return True

    def _get_header(self):
        """Get the header for the solver, with the name and options.

        Returns
        -------
        table : rich.Table
            Table with the solver name and options.
        """

        # Initialisation
        table = logging.Table(title="Options")
        table.add_column("Option", justify="right")
        table.add_column("Value", justify="right", style="option")

        def _check_modified(val, old):
            """Check if an option has been modified."""
            if type(val) is not type(old):
                return True
            if isinstance(val, np.ndarray):
                return not np.array_equal(val, old)
            return val != old

        # Loop over options
        for key, val in self._opts.items():
            if self._opt_is_used(key):
                if isinstance(val, dict):
                    # Format each entry of the dictionary
                    keys, vals = zip(*val.items()) if val else ((), ())
                    old = self.__class__._defaults.get(key, None)
                    keys = [f"{key}.{k}" for k in keys]
                    mods = [old and _check_modified(v, old[k]) for k, v in val.items()]
                else:
                    # Format the single value
                    keys = [key]
                    vals = [val]
                    mods = [_check_modified(val, self._defaults.get(key, None))]

                # Loop over entries
                for key, val, mod in zip(keys, vals, mods):
                    # Get the style for the key
                    key = f"[dim]{key}[/]" if mod else key

                    if isinstance(val, np.ndarray):
                        # Format numpy arrays
                        arr = np.array2string(
                            val,
                            precision=6,
                            separator=", ",
                            edgeitems=1,
                            threshold=0,
                        )
                        arr = f"np.array({arr})"
                        table.add_row(key, arr)
                    elif callable(val) or isinstance(val, type):
                        # Format functions and classes
                        table.add_row(key, val.__name__)
                    else:
                        # Format everything else using repr
                        table.add_row(key, repr(val))

        return table

    def _kernel(self, *args, **kwargs):
        """Abstract method for the kernel function."""
        raise NotImplementedError

    def kernel(self, *args, **kwargs):
        """Abstract method for the kernel driver function."""
        raise NotImplementedError

    def run(self, *args, **kwargs):
        """Alias for `kernel`, instead returning `self`.

        Parameters
        ----------
        *args : tuple
            Positional arguments to pass to `kernel`.
        **kwargs : dict
            Keyword arguments to pass to `kernel`.

        Returns
        -------
        self : BaseGW
            The solver object.
        """
        self.kernel(*args, **kwargs)
        return self

    @property
    def mol(self):
        """Get the molecule object."""
        return self._scf.mol

    @property
    def with_df(self):
        """Get the density fitting object."""
        if getattr(self._scf, "with_df", None) is None:
            raise ValueError("GW solvers require density fitting.")
        return self._scf.with_df

    @property
    def nao(self):
        """Get the number of atomic orbitals."""
        return self._scf.mol.nao

    @functools.cached_property
    def nmo(self):
        """Get the number of molecular orbitals."""
        frozen = self.frozen if self.frozen is not None else []
        if not isinstance(frozen, (list, np.ndarray)):
            raise ValueError("`frozen` must be a list or array of indices of orbitals to freeze.")
        occ = np.array(self._scf.mo_occ)
        nmo = np.full(occ.shape[:-1], fill_value=occ.shape[-1], dtype=int)
        nmo -= len(frozen)
        if np.isscalar(nmo):
            nmo = np.asarray(nmo).item()
        return nmo

    @functools.cached_property
    def nocc(self):
        """Get the number of occupied molecular orbitals."""
        frozen = self.frozen if self.frozen is not None else []
        if not isinstance(frozen, (list, np.ndarray)):
            raise ValueError("`frozen` must be a list or array of indices of orbitals to freeze.")
        occ = np.array(self._scf.mo_occ)
        nocc = np.sum(occ > 0, axis=-1)
        nocc -= sum(occ[..., i] > 0 for i in frozen)
        return nocc

    @functools.cached_property
    def active(self):
        """Get the mask to remove frozen orbitals."""
        frozen = self.frozen if self.frozen is not None else []
        if not isinstance(frozen, (list, np.ndarray)):
            raise ValueError("`frozen` must be a list or array of indices of orbitals to freeze.")
        nmo = np.array(self._scf.mo_occ).shape[-1]
        mask = np.ones((nmo,), dtype=bool)
        mask[frozen] = False
        return mask

    @property
    def frozen(self):
        """Get the frozen orbitals."""
        return self._frozen

    @frozen.setter
    def frozen(self, value):
        """Set the frozen orbitals."""
        if value is not None:
            self._frozen = np.asarray(value)
        del self.nmo, self.nocc, self.active

    @property
    def mo_energy(self):
        """Get the molecular orbital energies."""
        if self._mo_energy is None:
            self.mo_energy = self._scf.mo_energy
        return self._mo_energy[..., self.active]

    @property
    def mo_energy_with_frozen(self):
        """Get the molecular orbital energies with frozen orbitals."""
        if self._mo_energy is None:
            self.mo_energy = self._scf.mo_energy
        return self._mo_energy

    @mo_energy.setter
    def mo_energy(self, value):
        """Set the molecular orbital energies."""
        if value is not None:
            self._mo_energy = mpi_helper.bcast(np.asarray(value))
        del self.nmo, self.nocc, self.active

    @property
    def mo_coeff(self):
        """Get the molecular orbital coefficients."""
        if self._mo_coeff is None:
            self.mo_coeff = self._scf.mo_coeff
        return self._mo_coeff[..., self.active]

    @property
    def mo_coeff_with_frozen(self):
        """Get the molecular orbital coefficients with frozen orbitals."""
        if self._mo_coeff is None:
            self.mo_coeff = self._scf.mo_coeff
        return self._mo_coeff

    @mo_coeff.setter
    def mo_coeff(self, value):
        """Set the molecular orbital coefficients."""
        if value is not None:
            self._mo_coeff = mpi_helper.bcast(np.asarray(value))
        del self.nmo, self.nocc, self.active

    @property
    def mo_occ(self):
        """Get the molecular orbital occupation numbers."""
        if self._mo_occ is None:
            self.mo_occ = self._scf.mo_occ
        return self._mo_occ[..., self.active]

    @property
    def mo_occ_with_frozen(self):
        """
        Get the molecular orbital occupation numbers with frozen
        orbitals.
        """
        if self._mo_occ is None:
            self.mo_occ = self._scf.mo_occ
        return self._mo_occ

    @mo_occ.setter
    def mo_occ(self, value):
        """Set the molecular orbital occupation numbers."""
        if value is not None:
            self._mo_occ = mpi_helper.bcast(np.asarray(value))
        del self.nmo, self.nocc, self.active

    def __getattr__(self, key):
        """
        Try to get an attribute from the `_opts` dictionary. If it is
        not found, raise an `AttributeError`.

        Parameters
        ----------
        key : str
            Attribute key.

        Returns
        -------
        value : any
            Attribute value.
        """
        if key in self._defaults:
            return self._opts[key]
        raise AttributeError

    def __setattr__(self, key, val):
        """
        Try to set an attribute from the `_opts` dictionary. If it is
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


class BaseGW(Base):
    """Base class for moment-constrained GW solvers.

    Parameters
    ----------
    mf : pyscf.scf.SCF
        PySCF mean-field class.
    diagonal_se : bool, optional
        If `True`, use a diagonal approximation in the self-energy.
        Default value is `False`.
    polarizability : str, optional
        Type of polarizability to use, can be one of `("drpa",
        "drpa-exact", "dtda", "thc-dtda"). Default value is `"drpa"`.
    npoints : int, optional
        Number of numerical integration points. Default value is `48`.
    optimise_chempot : bool, optional
        If `True`, optimise the chemical potential by shifting the
        position of the poles in the self-energy relative to those in
        the Green's function. Default value is `False`.
    fock_loop : bool, optional
        If `True`, self-consistently renormalise the density matrix
        according to the updated Green's function. Default value is
        `False`.
    fock_opts : dict, optional
        Dictionary of options passed to the Fock loop. For more details
        see `momentGW.fock`.
    compression : str, optional
        Blocks of the ERIs to use as a metric for compression. Can be
        one or more of `("oo", "ov", "vv", "ia")` which can be passed as
        a comma-separated string. `"oo"`, `"ov"` and `"vv"` refer to
        compression on the initial ERIs, whereas `"ia"` refers to
        compression on the ERIs entering RPA, which may change under a
        self-consistent scheme. Default value is `"ia"`.
    compression_tol : float, optional
        Tolerance for the compression. Default value is `1e-10`.
    thc_opts : dict, optional
        Dictionary of options to be used for THC calculations. Current
        implementation requires a filepath to import the THC integrals.
    """

    _defaults = OrderedDict(
        diagonal_se=False,
        polarizability="drpa",
        npoints=48,
        optimise_chempot=False,
        fock_loop=False,
        fock_opts=OrderedDict(
            fock_diis_space=10,
            fock_diis_min_space=1,
            conv_tol_nelec=1e-6,
            conv_tol_rdm1=1e-8,
            max_cycle_inner=50,
            max_cycle_outer=20,
        ),
        compression="ia",
        compression_tol=1e-10,
        thc_opts=OrderedDict(
            file_path=None,
        ),
    )

    def __init__(self, mf, **kwargs):
        super().__init__(mf, **kwargs)

        # Attributes
        self.converged = None
        self.se = None
        self.gf = None
        self._qp_energy = None

    @property
    def name(self):
        """Abstract property for the solver name."""
        raise NotImplementedError

    def build_se_static(self, *args, **kwargs):
        """Abstract method for building the static self-energy."""
        raise NotImplementedError

    def build_se_moments(self, *args, **kwargs):
        """Abstract method for building the self-energy moments."""
        raise NotImplementedError

    def ao2mo(self, transform=True):
        """Abstract method for getting the integrals object."""
        raise NotImplementedError

    def solve_dyson(self, *args, **kwargs):
        """Abstract method for solving the Dyson equation."""
        raise NotImplementedError

    def _get_header(self):
        """
        Extend the header given by `Base._get_header` to include the
        problem size.

        Returns
        -------
        panel : rich.Table
            Panel with the solver name, options, and problem size.
        """

        # Get the options table
        options = super()._get_header()

        # Get the problem size table
        sizes = logging.Table(title="Sizes")
        sizes.add_column("Space", justify="right")
        sizes.add_column("Size", justify="right")
        sizes.add_row("MOs", f"{self.nmo}")
        sizes.add_row("Occupied MOs", f"{self.nocc}")
        sizes.add_row("Virtual MOs", f"{self.nmo - self.nocc}")

        # Combine the tables
        panel = logging.Table.grid()
        panel.add_row(options)
        panel.add_row("")
        panel.add_row(sizes)

        return panel

    def _get_energies_table(self, integrals):
        """Calculate the energies and return them as a table.

        Parameters
        ----------
        integrals : BaseIntegrals
            Integrals object.

        Returns
        -------
        table : rich.Table
            Table with the energies.
        """

        # Calculate energies
        e_1b_g0 = self._scf.e_tot
        e_1b = self.energy_hf(gf=self.gf, integrals=integrals) + self.energy_nuc()
        e_2b_g0 = self.energy_gm(se=self.se, g0=True)
        e_2b = self.energy_gm(gf=self.gf, se=self.se, g0=False)

        # Build table
        table = logging.Table(title="Energies")
        table.add_column("Functional", justify="right")
        table.add_column("Energy (G0)", justify="right", style="output")
        table.add_column("Energy (G)", justify="right", style="output")
        for name, e_g0, e_g in zip(
            ["One-body", "Galitskii-Migdal", "Total"],
            [e_1b_g0, e_2b_g0, e_1b_g0 + e_2b_g0],
            [e_1b, e_2b, e_1b + e_2b],
        ):
            table.add_row(name, f"{e_g0:.10f}", f"{e_g:.10f}")

        return table

    def _get_excitations_table(self):
        """Return the excitations as a table.

        Returns
        -------
        table : rich.Table
            Table with the excitations.
        """

        # Separate the occupied and virtual GFs
        gf_occ = self.gf.occupied().physical(weight=1e-1)
        gf_vir = self.gf.virtual().physical(weight=1e-1)

        # Build table
        table = logging.Table(title="Green's function poles")
        table.add_column("Excitation", justify="right")
        table.add_column("Energy", justify="right", style="output")
        table.add_column("QP weight", justify="right")
        table.add_column("Dominant MOs", justify="right")

        # Add IPs
        for n in range(min(3, gf_occ.naux)):
            en = -gf_occ.energies[-(n + 1)]
            weights = gf_occ.couplings[:, -(n + 1)] ** 2
            weight = np.sum(weights)
            dominant = np.argsort(weights)[::-1]
            dominant = dominant[weights[dominant] > 0.1][:3]
            mo_string = ", ".join([f"{i} ({100 * weights[i] / weight:5.1f}%)" for i in dominant])
            table.add_row(f"IP {n:>2}", f"{en:.10f}", f"{weight:.5f}", mo_string)

        # Add a break
        table.add_section()

        # Add EAs
        for n in range(min(3, gf_vir.naux)):
            en = gf_vir.energies[n]
            weights = gf_vir.couplings[:, n] ** 2
            weight = np.sum(weights)
            dominant = np.argsort(weights)[::-1]
            dominant = dominant[weights[dominant] > 0.1][:3]
            mo_string = ", ".join([f"{i} ({100 * weights[i] / weight:5.1f}%)" for i in dominant])
            table.add_row(f"EA {n:>2}", f"{en:.10f}", f"{weight:.5f}", mo_string)

        return table

    def _get_summary_panel(self, integrals, timer):
        """Return the summary as a panel.

        Parameters
        ----------
        integrals : BaseIntegrals
            Integrals object.
        timer : Timer
            Timer object.

        Returns
        -------
        panel : rich.Panel
            Panel with the summary.
        """

        # Get the convergence message
        if self.converged:
            msg = f"{self.name} [good]converged[/] in {timer.format_time(timer.total())}."
        else:
            msg = f"{self.name} [bad]did not converge[/] in {timer.format_time(timer.total())}."

        # Build the table
        table = logging._Table.grid()
        table.add_row(msg)
        table.add_row("")
        table.add_row(self._get_energies_table(integrals))
        table.add_row("")
        table.add_row(self._get_excitations_table())

        # Build the panel
        panel = logging.Panel(table, title="Summary", padding=(1, 2), expand=False)

        return panel

    def _convert_mf(self, mf):
        """Convert the mean-field object to the correct spin.

        Parameters
        ----------
        mf : pyscf.scf.SCF
            PySCF mean-field class.

        Returns
        -------
        mf : pyscf.scf.SCF
            PySCF mean-field class in the correct spin.
        """
        if hasattr(mf, "xc"):
            mf = mf.to_rks()
        else:
            mf = mf.to_rhf()
        return mf

    @logging.with_timer("Kernel")
    def kernel(
        self,
        nmom_max,
        moments=None,
        integrals=None,
    ):
        """Driver for the method.

        Parameters
        ----------
        nmom_max : int
            Maximum moment number to calculate.
        moments : tuple of numpy.ndarray, optional
            Tuple of (hole, particle) moments, if passed then they will
            be used instead of calculating them. Default value is
            `None`.
        integrals : BaseIntegrals, optional
            Integrals object. If `None`, generate from scratch. Default
            value is `None`.

        Returns
        -------
        converged : bool
            Whether the solver converged. For single-shot calculations,
            this is always `True`.
        gf : dyson.Lehmann
            Green's function object.
        se : dyson.Lehmann
            Self-energy object.
        qp_energy : numpy.ndarray
            Quasiparticle energies. For most GW methods, this is `None`.
        """

        # Start a timer
        timer = util.Timer()

        # Write the header
        logging.write("")
        logging.write(f"[bold underline]{self.name}[/]", comment="Solver options")
        logging.write("")
        logging.write(self._get_header())
        logging.write("", comment=f"Start of {self.name} kernel")
        logging.write(f"Solving for nmom_max = [option]{nmom_max}[/] ({nmom_max + 1} moments)")

        # Get the integrals
        if integrals is None:
            integrals = self.ao2mo()

        # Run the kernel
        with logging.with_status(f"Running {self.name} kernel"):
            self.converged, self.gf, self.se, self._qp_energy = self._kernel(
                nmom_max,
                integrals=integrals,
                moments=moments,
            )
        logging.write("", comment=f"End of {self.name} kernel")

        # Print the summary in a panel
        logging.write(self._get_summary_panel(integrals, timer))

        return self.converged, self.gf, self.se, self.qp_energy

    @staticmethod
    def _moment_error(t, t_prev):
        """Compute scaled error between moments.

        Parameters
        ----------
        t : list of numpy.ndarray
            List of moments.
        t_prev : list of numpy.ndarray
            List of previous moments.

        Returns
        -------
        error : float
            Maximum error between moments.
        """

        if t_prev is None:
            t_prev = np.zeros_like(t)

        error = 0
        for a, b in zip(t, t_prev):
            a = a / max(np.max(np.abs(a)), 1)
            b = b / max(np.max(np.abs(b)), 1)
            error = max(error, np.max(np.abs(a - b)))

        return error

    @staticmethod
    def _gf_to_occ(gf, occupancy=2):
        """
        Convert a `dyson.Lehmann` to an `mo_occ`.

        Parameters
        ----------
        gf : dyson.Lehmann
            Green's function object.
        occupancy : int, optional
            Number of electrons in each physical orbital. Default value
            is `2`.

        Returns
        -------
        occ : numpy.ndarray
            Orbital occupation numbers.
        """
        return gf.as_orbitals(occupancy=occupancy)[2]

    @staticmethod
    def _gf_to_energy(gf):
        """
        Convert a `dyson.Lehmann` to an `mo_energy`.

        Parameters
        ----------
        gf : dyson.Lehmann
            Green's function object.

        Returns
        -------
        energy : numpy.ndarray
            Orbital energies.
        """
        return gf.energies

    @staticmethod
    def _gf_to_coupling(gf, mo_coeff=None):
        """
        Convert a `dyson.Lehmann` to an `mo_coeff`.

        Parameters
        ----------
        gf : dyson.Lehmann
            Green's function object.
        mo_coeff : numpy.ndarray, optional
            Molecular orbital coefficients. If passed, rotate the
            Green's function couplings from the MO basis into the AO
            basis. Default value is `None`.

        Returns
        -------
        couplings : numpy.ndarray
            Couplings of the Green's function.
        """
        if mo_coeff is None:
            return gf.couplings
        else:
            return np.dot(mo_coeff, gf.couplings)

    def _gf_to_mo_energy(self, gf):
        """Find the poles of a GF which best overlap with the MOs.

        Parameters
        ----------
        gf : dyson.Lehmann
            Green's function object.

        Returns
        -------
        mo_energy : numpy.ndarray
            Updated MO energies.
        """

        # Initialise arrays
        check = set()
        best_weights = np.zeros((gf.nphys,))
        best_assignments = np.zeros((gf.nphys,), dtype=int)
        mo_energy = np.zeros((gf.nphys,))

        todo = set(range(gf.nphys))
        while todo:
            # Get the next index
            i = todo.pop()

            # Get the weights on the ith orbital for each pole
            weights = gf.couplings[i] * gf.couplings[i].conj()

            while True:
                # Get the pole with the largest weight
                arg = np.argmax(weights)

                # If the pole is already assigned, check if the current
                # assignment is better. If it is, add the assigned state
                # back to the todo list, otherwise get the next best
                if arg in check:
                    if weights[arg] > best_weights[i]:
                        todo.add(best_assignments[i])
                    else:
                        weights[arg] = 0
                        continue
                break

            # Assign the pole to the orbital
            check.add(arg)
            best_weights[i] = weights[arg]
            best_assignments[i] = arg
            mo_energy[i] = gf.energies[arg]

        return mo_energy

    @property
    def qp_energy(self):
        """
        Get the quasiparticle energies.

        Notes
        -----
        For most GW methods, this simply consists of the poles of the
        `self.gf` that best overlap with the MOs, in order. In some
        methods such as qsGW, these two quantities are not the same.
        """

        if self._qp_energy is not None:
            return self._qp_energy

        qp_energy = self._gf_to_mo_energy(self.gf)

        return qp_energy

    @property
    def has_fock_loop(self):
        """
        Get a boolean indicating whether the solver requires a Fock
        loop.

        Notes
        -----
        For most GW methods, this is simply `self.fock_loop`. In some
        methods such as qsGW, a Fock loop is required with or without
        `self.fock_loop` for the quasiparticle self-consistency, with
        this property acting as a hook to indicate this.
        """
        return self.fock_loop


class BaseSE:
    """Base class for computing self-energy moments.

    Parameters
    ----------
    gw : BaseGW
        GW object.
    nmom_max : int
        Maximum moment number to calculate.
    integrals : Integrals
        Integrals object.
    mo_energy : dict, optional
        Molecular orbital energies. Keys are "g" and "w" for the Green's
        function and screened Coulomb interaction, respectively.
        If `None`, use `gw.mo_energy` for both. Default value is `None`.
    mo_occ : dict, optional
        Molecular orbital occupancies. Keys are "g" and "w" for the
        Green's function and screened Coulomb interaction, respectively.
        If `None`, use `gw.mo_occ` for both. Default value is `None`.
    """

    def __init__(
        self,
        gw,
        nmom_max,
        integrals,
        mo_energy=None,
        mo_occ=None,
    ):
        # Attributes
        self.gw = gw
        self.nmom_max = nmom_max
        self.integrals = integrals
        self.mo_energy = mo_energy
        self.mo_occ = mo_occ

    def kernel(self):
        """
        Run the polarizability calculation to compute moments of the
        self-energy.

        Returns
        -------
        moments_occ : numpy.ndarray
            Moments of the occupied self-energy.
        moments_vir : numpy.ndarray
            Moments of the virtual self-energy.
        """
        raise NotImplementedError

    @property
    def nmo(self):
        """Get the number of MOs."""
        return self.gw.nmo

    @property
    def naux(self):
        """Get the number of auxiliaries."""
        return self.integrals.naux

    @functools.cached_property
    def nov(self):
        """
        Get the number of ov states in the screened Coulomb interaction.
        """
        return np.sum(self.mo_occ_w > 0) * np.sum(self.mo_occ_w == 0)

    def mpi_slice(self, n):
        """
        Return the start and end index for the current process for total
        size `n`.

        Parameters
        ----------
        n : int
            Total size.

        Returns
        -------
        p0 : int
            Start index for current process.
        p1 : int
            End index for current process.
        """
        return list(mpi_helper.prange(0, n, n))[0]

    def mpi_size(self, n):
        """
        Return the number of states in the current process for total size
        `n`.

        Parameters
        ----------
        n : int
            Total size.

        Returns
        -------
        size : int
            Number of states in current process.
        """
        p0, p1 = self.mpi_slice(n)
        return p1 - p0
