"""
Base class for moment-constained GW solvers with unrestricted
references.
"""

import numpy as np
from pyscf.mp.ump2 import get_frozen_mask, get_nmo, get_nocc

from momentGW import logging
from momentGW.base import Base, BaseGW


class BaseUGW(BaseGW):
    """
    Base class for moment-constrained GW solvers with unrestricted
    references.

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

    get_nmo = get_nmo
    get_nocc = get_nocc
    get_frozen_mask = get_frozen_mask

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
        options = Base._get_header(self)

        # Get the problem size table
        sizes = logging.Table(title="Sizes")
        sizes.add_column("Space", justify="right")
        sizes.add_column("Size (α)", justify="right")
        sizes.add_column("Size (β)", justify="right")
        sizes.add_row("MOs", f"{self.nmo[0]}", f"{self.nmo[1]}")
        sizes.add_row("Occupied MOs", f"{self.nocc[0]}", f"{self.nocc[1]}")
        sizes.add_row(
            "Virtual MOs", f"{self.nmo[0] - self.nocc[0]}", f"{self.nmo[1] - self.nocc[1]}"
        )

        # Combine the tables
        panel = logging.Table.grid()
        panel.add_row(options)
        panel.add_row("")
        panel.add_row(sizes)

        return panel

    def _get_excitations_table(self):
        """Return the excitations as a table.

        Returns
        -------
        table : rich.Table
            Table with the excitations.
        """

        # Separate the occupied and virtual GFs
        gf_occ = (
            self.gf[0].occupied().physical(weight=1e-1),
            self.gf[1].occupied().physical(weight=1e-1),
        )
        gf_vir = (
            self.gf[0].virtual().physical(weight=1e-1),
            self.gf[1].virtual().physical(weight=1e-1),
        )

        # Build table
        table = logging.Table(title="Green's function poles")
        table.add_column("Excitation", justify="right")
        table.add_column("Energy", justify="right", style="output")
        table.add_column("QP weight", justify="right")
        table.add_column("Dominant MOs", justify="right")

        # Add IPs
        for s, spin in enumerate(["α", "β"]):
            for n in range(min(3, gf_occ[s].naux)):
                en = -gf_occ[s].energies[-(n + 1)]
                weights = gf_occ[s].couplings[:, -(n + 1)] ** 2
                weight = np.sum(weights)
                dominant = np.argsort(weights)[::-1]
                dominant = dominant[weights[dominant] > 0.1][:3]
                mo_string = ", ".join(
                    [f"{i} ({100 * weights[i] / weight:5.1f}%)" for i in dominant]
                )
                table.add_row(f"IP ({spin}) {n:>2}", f"{en:.10f}", f"{weight:.5f}", mo_string)

            # Add a break
            table.add_section()

        for s, spin in enumerate(["α", "β"]):
            # Add EAs
            for n in range(min(3, gf_vir[s].naux)):
                en = gf_vir[s].energies[n]
                weights = gf_vir[s].couplings[:, n] ** 2
                weight = np.sum(weights)
                dominant = np.argsort(weights)[::-1]
                dominant = dominant[weights[dominant] > 0.1][:3]
                mo_string = ", ".join(
                    [f"{i} ({100 * weights[i] / weight:5.1f}%)" for i in dominant]
                )
                table.add_row(f"EA ({spin}) {n:>2}", f"{en:.10f}", f"{weight:.5f}", mo_string)

            # Add a break
            if s != 1:
                table.add_section()

        return table

    @staticmethod
    def _gf_to_occ(gf):
        """
        Convert a `dyson.Lehmann` to an `mo_occ`.

        Parameters
        ----------
        gf : tuple of dyson.Lehmann
            Green's function object for each spin channel.

        Returns
        -------
        occ : tuple of numpy.ndarray
            Orbital occupation numbers for each spin channel.
        """
        return tuple(BaseGW._gf_to_occ(g, occupancy=1) for g in gf)

    @staticmethod
    def _gf_to_energy(gf):
        """
        Convert a `dyson.Lehmann` to an `mo_energy`.

        Parameters
        ----------
        gf : tuple of dyson.Lehmann
            Green's function object for each spin channel.

        Returns
        -------
        energy : tuple of numpy.ndarray
            Orbital energies for each spin channel.
        """
        return tuple(BaseGW._gf_to_energy(g) for g in gf)

    @staticmethod
    def _gf_to_coupling(gf, mo_coeff=None):
        """
        Convert a `dyson.Lehmann` to an `mo_coeff`.

        Parameters
        ----------
        gf : tuple of dyson.Lehmann
            Green's function object for each spin channel.
        mo_coeff : tuple of numpy.ndarray, optional
            Molecular orbital coefficients for each spin channel. If
            passed, rotate the Green's function couplings from the MO
            basis into the AO basis. Default value is `None`.

        Returns
        -------
        couplings : tuple of numpy.ndarray
            Couplings of the Green's function for each spin channel.
        """
        if mo_coeff is None:
            mo_coeff = [None] * 2
        return tuple(BaseGW._gf_to_coupling(g, mo) for g, mo in zip(gf, mo_coeff))

    def _gf_to_mo_energy(self, gf):
        """Find the poles of a GF which best overlap with the MOs.

        Parameters
        ----------
        gf : tuple of dyson.Lehmann
            Green's function object for each spin channel.

        Returns
        -------
        mo_energy : numpy.ndarray
            Updated MO energies for each spin channel.
        """
        return np.array([BaseGW._gf_to_mo_energy(self, g) for g in gf])
