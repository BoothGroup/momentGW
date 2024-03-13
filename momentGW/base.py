"""
Base class for moment-constrained GW solvers.
"""

import numpy as np
from pyscf.mp.mp2 import get_frozen_mask, get_nmo, get_nocc

from momentGW import init_logging, logging, mpi_helper, util


class Base:
    """Base class."""

    _opts = []

    def __init__(
        self,
        mf,
        mo_energy=None,
        mo_coeff=None,
        mo_occ=None,
        **kwargs,
    ):
        # Parameters
        self._scf = mf
        self._mo_energy = mpi_helper.bcast(np.asarray(mo_energy)) if mo_energy is not None else None
        self._mo_coeff = mpi_helper.bcast(np.asarray(mo_coeff)) if mo_coeff is not None else None
        self._mo_occ = mpi_helper.bcast(np.asarray(mo_occ)) if mo_occ is not None else None
        self._nmo = None
        self._nocc = None
        self.frozen = None

        # Options
        for key, val in kwargs.items():
            if not hasattr(self, key):
                raise AttributeError(f"{key} is not a valid option for {self.name}")
            setattr(self, key, val)

        # Logging
        init_logging()
        logging.write("")
        logging.write("[bold underline]{self.name}[/]", comment=f"Initialisation of solver")
        table = logging.Table(title="Options")
        table.add_column("Option", justify="right")
        table.add_column("Value", justify="right", style="option")
        for key in self._opts:
            if self._opt_is_used(key):
                val = getattr(self, key)
                if isinstance(val, dict):
                    keys, vals = zip(*val.items())
                    keys = [f"{key}.{k}" for k in keys]
                else:
                    keys = [key]
                    vals = [val]
                for key, val in zip(keys, vals):
                    if isinstance(val, np.ndarray):
                        arr = np.array2string(
                            val,
                            precision=6,
                            separator=", ",
                            edgeitems=1,
                            threshold=0,
                        )
                        arr = f"np.array({arr})"
                        table.add_row(key, arr)
                    else:
                        table.add_row(key, repr(val))
        logging.write("")
        logging.write(table)

    def _kernel(self, *args, **kwargs):
        raise NotImplementedError

    def _opt_is_used(self, key):
        """
        Check if an option is used by the solver. This is useful for
        determining whether to print the option in the table.
        """
        if key == "fock_opts":
            return self.fock_loop
        if key == "thc_opts":
            return self.polarizability.lower().startswith("thc")
        if key == "npoints":
            return self.polarizability.lower().endswith("drpa")
        return True

    @property
    def mol(self):
        """Molecule object."""
        return self._scf.mol

    @property
    def with_df(self):
        """Density fitting object."""
        if getattr(self._scf, "with_df", None) is None:
            raise ValueError("GW solvers require density fitting.")
        return self._scf.with_df

    get_nmo = get_nmo
    get_nocc = get_nocc
    get_frozen_mask = get_frozen_mask

    @property
    def nmo(self):
        """Number of molecular orbitals."""
        return self.get_nmo()

    @property
    def nocc(self):
        """Number of occupied molecular orbitals."""
        return self.get_nocc()

    @property
    def mo_energy(self):
        """Molecular orbital energies."""
        if self._mo_energy is None:
            return self._scf.mo_energy
        return self._mo_energy

    @property
    def mo_coeff(self):
        """Molecular orbital coefficients."""
        if self._mo_coeff is None:
            return self._scf.mo_coeff
        return self._mo_coeff

    @property
    def mo_occ(self):
        """Molecular orbital occupation numbers."""
        if self._mo_occ is None:
            return self._scf.mo_occ
        return self._mo_occ


class BaseGW(Base):
    """{description}

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
    {extra_parameters}
    """

    # --- Default GW options

    diagonal_se = False
    polarizability = "drpa"
    npoints = 48
    optimise_chempot = False
    fock_loop = False
    fock_opts = dict(
        fock_diis_space=10,
        fock_diis_min_space=1,
        conv_tol_nelec=1e-6,
        conv_tol_rdm1=1e-8,
        max_cycle_inner=50,
        max_cycle_outer=20,
    )
    compression = "ia"
    compression_tol = 1e-10
    thc_opts = dict(
        file_path=None,
    )

    _opts = [
        "diagonal_se",
        "polarizability",
        "npoints",
        "optimise_chempot",
        "fock_loop",
        "fock_opts",
        "compression",
        "compression_tol",
        "thc_opts",
    ]

    def __init__(self, mf, **kwargs):
        super().__init__(mf, **kwargs)

        # Attributes
        self.converged = None
        self.se = None
        self.gf = None
        self._qp_energy = None

    def build_se_static(self, *args, **kwargs):
        """Abstract method for building the static self-energy."""
        raise NotImplementedError

    def build_se_moments(self, *args, **kwargs):
        """Abstract method for building the self-energy moments."""
        raise NotImplementedError

    def solve_dyson(self, *args, **kwargs):
        """Abstract method for solving the Dyson equation."""
        raise NotImplementedError

    @logging.with_timer("Kernel")
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
            Molecular orbital energies.
        mo_coeff : numpy.ndarray
            Molecular orbital coefficients.
        moments : tuple of numpy.ndarray, optional
            Tuple of (hole, particle) moments, if passed then they will
            be used instead of calculating them. Default value is
            `None`.
        integrals : Integrals, optional
            Integrals object. If `None`, generate from scratch. Default
            value is `None`.
        """

        timer = util.Timer()

        if mo_coeff is None:
            mo_coeff = self.mo_coeff
        if mo_energy is None:
            mo_energy = self.mo_energy

        logging.write("")
        logging.write(f"Solving for nmom_max = [option]{nmom_max}[/] ({nmom_max + 1} moments)")

        if integrals is None:
            integrals = self.ao2mo()

        with logging.with_status(f"Running {self.name} kernel"):
            logging.write("", comment=f"Start of {self.name} kernel")
            self.converged, self.gf, self.se, self._qp_energy = self._kernel(
                nmom_max,
                mo_energy,
                mo_coeff,
                integrals=integrals,
                moments=moments,
            )
            logging.write("", comment=f"End of {self.name} kernel")

        # Print the summary in a panel
        logging.write(self._get_summary_panel(integrals, timer))

        return self.converged, self.gf, self.se, self.qp_energy

    def run(self, *args, **kwargs):
        """Alias for `kernel`, instead returning `self`."""
        self.kernel(*args, **kwargs)
        return self

    @staticmethod
    def _moment_error(t, t_prev):
        """Compute scaled error between moments."""

        if t_prev is None:
            t_prev = np.zeros_like(t)

        error = 0
        for a, b in zip(t, t_prev):
            a = a / max(np.max(np.abs(a)), 1)
            b = b / max(np.max(np.abs(b)), 1)
            error = max(error, np.max(np.abs(a - b)))

        return error

    def _get_summary_panel(self, integrals, timer):
        """Return the summary as a panel."""

        if self.converged:
            msg = f"{self.name} [good]converged[/] in {timer.format_time(timer.total())}."
        else:
            msg = f"{self.name} [bad]did not converge[/] in {timer.format_time(timer.total())}."

        table = logging._Table.grid()
        table.add_row(msg)
        table.add_row("")
        table.add_row(self._get_energies_table(integrals))
        table.add_row("")
        table.add_row(self._get_excitations_table())

        panel = logging.Panel(table, title="Summary", padding=(1, 2), expand=False)

        return panel

    def _get_energies_table(self, integrals):
        """Calculate the energies and return them as a table."""

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
        """Return the excitations as a table."""

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
        for n in range(min(5 if logging.level >= 2 else 3, gf_occ.naux)):
            en = -gf_occ.energies[-(n + 1)]
            weights = gf_occ.couplings[:, -(n + 1)] ** 2
            weight = np.sum(weights)
            dominant = np.argsort(weights)[::-1]
            dominant = dominant[weights[dominant] > 0.1][:3]
            mo_string = ", ".join([f"{i} ({100 * weights[i] / weight:5.1f}%)" for i in dominant])
            table.add_row(f"IP {n:>2}", f"{en:.10f}", f"{weight:.5f}", mo_string)

        # Add EAs
        for n in range(min(5 if logging.level >= 2 else 3, gf_vir.naux)):
            en = gf_vir.energies[n]
            weights = gf_vir.couplings[:, n] ** 2
            weight = np.sum(weights)
            dominant = np.argsort(weights)[::-1]
            dominant = dominant[weights[dominant] > 0.1][:3]
            mo_string = ", ".join([f"{i} ({100 * weights[i] / weight:5.1f}%)" for i in dominant])
            table.add_row(f"EA {n:>2}", f"{en:.10f}", f"{weight:.5f}", mo_string)

        return table

    @staticmethod
    def _gf_to_occ(gf, occupancy=2):
        """Convert a `dyson.Lehmann` to an `mo_occ`. Allows hooking in
        `pbc` methods to retain syntax.
        """
        return gf.as_orbitals(occupancy=occupancy)[2]

    @staticmethod
    def _gf_to_energy(gf):
        """
        Return the `energy` attribute of a `gf`. Allows hooking in `pbc`
        methods to retain syntax.
        """
        return gf.energies

    @staticmethod
    def _gf_to_coupling(gf, mo_coeff=None):
        """
        Return the `coupling` attribute of a `gf`. Allows hooking in
        `pbc` methods to retain syntax.
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
        mo_energy : ndarray
            Updated MO energies.
        """

        check = set()
        mo_energy = np.zeros((gf.nphys,))

        for i in range(gf.nphys):
            arg = np.argmax(gf.couplings[i] ** 2)
            mo_energy[i] = gf.energies[arg]
            check.add(arg)

        if len(check) != gf.nphys:
            # TODO improve this warning
            logging.warn("[bad]Inconsistent quasiparticle weights![/]")

        return mo_energy

    @property
    def qp_energy(self):
        """
        Return the quasiparticle energies. For most GW methods, this
        simply consists of the poles of the `self.gf` that best
        overlap with the MOs, in order. In some methods such as qsGW,
        these two quantities are not the same.
        """

        if self._qp_energy is not None:
            return self._qp_energy

        qp_energy = self._gf_to_mo_energy(self.gf)

        return qp_energy

    @property
    def has_fock_loop(self):
        """
        Returns a boolean indicating whether the solver requires a Fock
        loop. For most GW methods, this is simply `self.fock_loop`. In
        some methods such as qsGW, a Fock loop is required with or
        without `self.fock_loop` for the quasiparticle self-consistency,
        with this property acting as a hook to indicate this.
        """
        return self.fock_loop
