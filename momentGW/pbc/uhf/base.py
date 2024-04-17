"""
Base class for moment-constrained GW solvers with periodic boundary
conditions and unrestricted references.
"""

import numpy as np

from momentGW import logging
from momentGW.base import Base, BaseGW
from momentGW.pbc.base import BaseKGW
from momentGW.uhf.base import BaseUGW


class BaseKUGW(BaseKGW, BaseUGW):
    """
    Base class for moment-constrained GW solvers for periodic systems
    with unrestricted references.

    Parameters
    ----------
    mf : pyscf.pbc.scf.KSCF
        PySCF periodic mean-field class.
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
        see `momentGW.pbc.fock`.
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
    fc : bool, optional
        If `True`, apply finite size corrections. Default value is
        `False`.
    """

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
        sizes.add_column("Size (Γ, α)", justify="right")
        sizes.add_column("Size (Γ, β)", justify="right")
        sizes.add_row("MOs", f"{self.nmo[0]}", f"{self.nmo[1]}")
        sizes.add_row("Occupied MOs", f"{self.nocc[0][0]}", f"{self.nocc[1][0]}")
        sizes.add_row(
            "Virtual MOs", f"{self.nmo[0] - self.nocc[0][0]}", f"{self.nmo[1] - self.nocc[1][0]}"
        )
        sizes.add_row("k-points", f"{self.kpts.kmesh} = {self.nkpts}")

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
            self.gf[0][0].occupied().physical(weight=1e-1),
            self.gf[1][0].occupied().physical(weight=1e-1),
        )
        gf_vir = (
            self.gf[0][0].virtual().physical(weight=1e-1),
            self.gf[1][0].virtual().physical(weight=1e-1),
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
                weights = np.real(
                    gf_occ[s].couplings[:, -(n + 1)] * gf_occ[s].couplings[:, -(n + 1)].conj()
                )
                weight = np.sum(weights)
                dominant = np.argsort(weights)[::-1]
                dominant = dominant[weights[dominant] > 0.1][:3]
                mo_string = ", ".join(
                    [f"{i} ({100 * weights[i] / weight:5.1f}%)" for i in dominant]
                )
                table.add_row(f"IP (Γ, {spin}) {n:>2}", f"{en:.10f}", f"{weight:.5f}", mo_string)

            # Add a break
            table.add_section()

        for s, spin in enumerate(["α", "β"]):
            # Add EAs
            for n in range(min(3, gf_vir[s].naux)):
                en = gf_vir[s].energies[n]
                weights = np.real(gf_vir[s].couplings[:, n] * gf_vir[s].couplings[:, n].conj())
                weight = np.sum(weights)
                dominant = np.argsort(weights)[::-1]
                dominant = dominant[weights[dominant] > 0.1][:3]
                mo_string = ", ".join(
                    [f"{i} ({100 * weights[i] / weight:5.1f}%)" for i in dominant]
                )
                table.add_row(f"EA (Γ, {spin}) {n:>2}", f"{en:.10f}", f"{weight:.5f}", mo_string)

            # Add a break
            if s != 1:
                table.add_section()

        return table

    @staticmethod
    def _gf_to_occ(gf):
        """
        Convert a `dyson.Lehmann` to an `mo_occ` at each k-point for
        each spin channel.

        Parameters
        ----------
        gf : tuple of tuple of dyson.Lehmann
            Green's function object at each k-point for each spin
            channel.

        Returns
        -------
        occ : tuple of tuple of numpy.ndarray
            Orbital occupation numbers at each k-point for each spin
            channel.
        """
        return tuple(tuple(BaseGW._gf_to_occ(g, occupancy=1) for g in gs) for gs in gf)

    @staticmethod
    def _gf_to_energy(gf):
        """
        Convert a `dyson.Lehmann` to an `mo_energy` for each spin
        channel.

        Parameters
        ----------
        gf : tuple of tuple of dyson.Lehmann
            Green's function object at each k-point for each spin
            channel.

        Returns
        -------
        energy : tuple of tuple of numpy.ndarray
            Orbital energies at each k-point for each spin channel.
        """
        return tuple(tuple(BaseGW._gf_to_energy(g) for g in gs) for gs in gf)

    @staticmethod
    def _gf_to_coupling(gf, mo_coeff=None):
        """
        Convert a `dyson.Lehmann` to an `mo_coeff`.

        Parameters
        ----------
        gf : tuple of tuple of dyson.Lehmann
            Green's function object at each k-point for each spin
            channel.
        mo_coeff : tuple of tuple of numpy.ndarray, optional
            Molecular orbital coefficients at each k-point for each
            spin channel. If passed, rotate the Green's function
            couplings from the MO basis into the AO basis. Default
            value is `None`.

        Returns
        -------
        couplings : tuple of tuple of numpy.ndarray
            Couplings of the Green's function at each k-point for each
            spin channel.
        """
        if mo_coeff is None:
            mo_coeff = [[None] * len(gf[0])] * 2
        return tuple(
            tuple(BaseGW._gf_to_coupling(g, mo) for g, mo in zip(gs, mos))
            for gs, mos in zip(gf, mo_coeff)
        )

    def _gf_to_mo_energy(self, gf):
        """Find the poles of a GF which best overlap with the MOs.

        Parameters
        ----------
        gf : tuple of tuple of dyson.Lehmann
            Green's function object at each k-point for each spin
            channel.

        Returns
        -------
        mo_energy : numpy.ndarray
            Updated MO energies at each k-point for each spin channel.
        """
        return np.array([[BaseGW._gf_to_mo_energy(self, g) for g in gs] for gs in gf])
