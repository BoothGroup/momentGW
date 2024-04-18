"""
Spin-unrestricted one-shot GW via self-energy moment constraints for
molecular systems.
"""

import numpy as np
from dyson import MBLSE, Lehmann, MixedMBLSE

from momentGW import energy, logging, util
from momentGW.fock import search_chempot
from momentGW.gw import GW
from momentGW.uhf.base import BaseUGW
from momentGW.uhf.fock import FockLoop
from momentGW.uhf.ints import UIntegrals
from momentGW.uhf.rpa import dRPA
from momentGW.uhf.tda import dTDA


class UGW(BaseUGW, GW):
    """
    Spin-unrestricted one-shot GW via self-energy moment constraints for
    molecules.

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

    Notes
    -----
    This approach is described in [1]_.

    References
    ----------
    .. [1] C. J. C. Scott, O. J. Backhouse, and G. H. Booth, 158, 12,
        2023.
    """

    @property
    def name(self):
        """Get the method name."""
        return f"{self.polarizability_name}-UG0W0"

    @logging.with_timer("Static self-energy")
    @logging.with_status("Building static self-energy")
    def build_se_static(self, integrals):
        """
        Build the static part of the self-energy, including the Fock
        matrix.

        Parameters
        ----------
        integrals : UIntegrals
            Integrals object.

        Returns
        -------
        se_static : numpy.ndarray
            Static part of the self-energy for each spin channel. If
            `self.diagonal_se`, non-diagonal elements are set to zero.
        """
        return super().build_se_static(integrals)

    def build_se_moments(self, nmom_max, integrals, **kwargs):
        """Build the moments of the self-energy.

        Parameters
        ----------
        nmom_max : int
            Maximum moment number to calculate.
        integrals : UIntegrals
            Integrals object.
        **kwargs : dict, optional
           Additional keyword arguments passed to polarizability class.

        Returns
        -------
        se_moments_hole : tuple of numpy.ndarray
            Moments of the hole self-energy for each spin channel. If
            `self.diagonal_se`, non-diagonal elements are set to zero.
        se_moments_part : tuple of numpy.ndarray
            Moments of the particle self-energy for each spin channel.
            If `self.diagonal_se`, non-diagonal elements are set to
            zero.

        See Also
        --------
        momentGW.uhf.rpa.dRPA
        momentGW.uhf.tda.dTDA
        """

        if self.polarizability.lower() == "dtda":
            tda = dTDA(self, nmom_max, integrals, **kwargs)
            return tda.kernel()

        elif self.polarizability.lower() == "drpa":
            rpa = dRPA(self, nmom_max, integrals, **kwargs)
            return rpa.kernel()

        else:
            raise NotImplementedError

    @logging.with_timer("Integral construction")
    @logging.with_status("Constructing integrals")
    def ao2mo(self, transform=True):
        """Get the integrals object.

        Parameters
        ----------
        transform : bool, optional
            Whether to transform the integrals object.

        Returns
        -------
        integrals : UIntegrals
            Integrals object.

        See Also
        --------
        momentGW.uhf.ints.UIntegrals
        """

        # Get the integrals
        integrals = UIntegrals(
            self.with_df,
            self.mo_coeff,
            self.mo_occ,
            compression=self.compression,
            compression_tol=self.compression_tol,
            store_full=self.fock_loop,
        )

        # Transform the integrals
        if transform:
            integrals.transform()

        return integrals

    def solve_dyson(self, se_moments_hole, se_moments_part, se_static, integrals=None):
        """
        Solve the Dyson equation due to a self-energy resulting from a
        list of hole and particle moments, along with a static
        contribution for each spin channel.

        Also finds a chemical potential best satisfying the physical
        number of electrons. If `self.optimise_chempot`, this will
        shift the self-energy poles relative to the Green's function,
        which is a partial self-consistency that better conserves the
        particle number.

        If `self.fock_loop`, this function will also require that the
        outputted Green's function is self-consistent with respect to
        the corresponding density and Fock matrix.

        Parameters
        ----------
        se_moments_hole : tuple of numpy.ndarray
            Moments of the hole self-energy for each spin channel.
        se_moments_part : tuple of numpy.ndarray
            Moments of the particle self-energy for each spin channel.
        se_static : tuple of numpy.ndarray
            Static part of the self-energy for each spin channel.
        integrals : UIntegrals
            Integrals object. Required if `self.fock_loop` is `True`.
            Default value is `None`.

        Returns
        -------
        gf : tuple of dyson.Lehmann
            Green's function for each spin channel.
        se : tuple of dyson.Lehmann
            Self-energy for each spin channel.

        See Also
        --------
        momentGW.uhf.fock.FockLoop
        """

        # Solve the Dyson equation for the moments
        with logging.with_modifiers(status="Solving Dyson equation", timer="Dyson equation"):
            solver_occ = MBLSE(se_static[0], np.array(se_moments_hole[0]))
            solver_occ.kernel()

            solver_vir = MBLSE(se_static[0], np.array(se_moments_part[0]))
            solver_vir.kernel()

            solver = MixedMBLSE(solver_occ, solver_vir)
            se_α = solver.get_self_energy()

            solver_occ = MBLSE(se_static[1], np.array(se_moments_hole[1]))
            solver_occ.kernel()

            solver_vir = MBLSE(se_static[1], np.array(se_moments_part[1]))
            solver_vir.kernel()

            solver = MixedMBLSE(solver_occ, solver_vir)
            se_β = solver.get_self_energy()

            se = (se_α, se_β)

        # Initialise the solver
        solver = FockLoop(self, se=se, **self.fock_opts)

        # Shift the self-energy poles relative to the Green's function
        # to better conserve the particle number
        if self.optimise_chempot:
            se = solver.auxiliary_shift(se_static)

        # Find the error in the moments
        error = (
            self.moment_error(se_moments_hole[0], se_moments_part[0], se[0]),
            self.moment_error(se_moments_hole[1], se_moments_part[1], se[1]),
        )
        for s, spin in enumerate(["α", "β"]):
            logging.write(
                f"Error in moments ({spin}):  "
                f"[{logging.rate(sum(error[s]), 1e-12, 1e-8)}]{sum(error[s]):.3e}[/] "
                f"(hole = [{logging.rate(error[s][0], 1e-12, 1e-8)}]{error[s][0]:.3e}[/], "
                f"particle = [{logging.rate(error[s][1], 1e-12, 1e-8)}]{error[s][1]:.3e}[/])"
            )

        # Solve the Dyson equation for the self-energy
        gf, error = solver.solve_dyson(se_static)
        se[0].chempot = gf[0].chempot
        se[1].chempot = gf[1].chempot

        # Self-consistently renormalise the density matrix
        if self.fock_loop:
            logging.write("")
            solver.gf = gf
            solver.se = se
            conv, gf, se = solver.kernel(integrals=integrals)
            _, error = solver.search_chempot(gf)

        # Print the error in the number of electrons
        logging.write("")
        color = logging.rate(
            abs(error),
            1e-6,
            1e-6 if self.fock_loop or self.optimise_chempot else 1e-1,
        )
        logging.write(f"Error in number of electrons:  [{color}]{error:.3e}[/]")
        for s, spin in enumerate(["α", "β"]):
            logging.write(f"Chemical potential ({spin}):  {gf[s].chempot:.6f}")

        return tuple(gf), tuple(se)

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
            Tuple of (hole, particle) moments for each spin channel, if
            passed then they will be used instead of calculating them.
            Default value is `None`.
        integrals : UIntegrals, optional
            Integrals object. If `None`, generate from scratch. Default
            value is `None`.

        Returns
        -------
        converged : bool
            Whether the solver converged. For single-shot calculations,
            this is always `True`.
        gf : tuple of dyson.Lehmann
            Green's function object for each spin channel.
        se : tuple of dyson.Lehmann
            Self-energy object for each spin channel.
        qp_energy : NoneType
            Quasiparticle energies. For most GW methods, this is `None`.
        """
        return super().kernel(nmom_max, moments=moments, integrals=integrals)

    def make_rdm1(self, gf=None):
        """Get the first-order reduced density matrix.

        Parameters
        ----------
        gf : tuple of dyson.Lehmann, optional
            Green's function for each spin channel. If `None`, use
            either `self.gf`, or the mean-field Green's function.
            Default value is `None`.

        Returns
        -------
        rdm1 : numpy.ndarray
            First-order reduced density matrix for each spin channel.
        """

        if gf is None:
            gf = self.gf
        if gf is None:
            gf = self.init_gf()

        return (gf[0].occupied().moment(0), gf[1].occupied().moment(0))

    @logging.with_timer("Energy")
    @logging.with_status("Calculating energy")
    def energy_hf(self, gf=None, integrals=None):
        """Calculate the one-body (Hartree--Fock) energy.

        Parameters
        ----------
        gf : tuple of dyson.Lehmann, optional
            Green's function for each spin channel. If `None`, use
            either `self.gf`, or the mean-field Green's function.
            Default value is `None`.
        integrals : UIntegrals, optional
            Integrals object. If `None`, generate from scratch. Default
            value is `None`.

        Returns
        -------
        e_1b : float
            One-body energy.
        """

        # Get the Green's function
        if gf is None:
            gf = self.gf

        # Get the integrals
        if integrals is None:
            integrals = self.ao2mo()

        # Form the Fock matrix
        with util.SilentSCF(self._scf):
            h1e = tuple(
                util.einsum("pq,pi,qj->ij", self._scf.get_hcore(), c.conj(), c)
                for c in self.mo_coeff
            )
        rdm1 = self.make_rdm1(gf=gf)
        fock = integrals.get_fock(rdm1, h1e)

        # Calculate the energy parts
        e_1b = energy.hartree_fock(rdm1[0], fock[0], h1e[0])
        e_1b += energy.hartree_fock(rdm1[1], fock[1], h1e[1])

        return e_1b

    @logging.with_timer("Energy")
    @logging.with_status("Calculating energy")
    def energy_gm(self, gf=None, se=None, g0=True):
        r"""Calculate the two-body (Galitskii--Migdal) energy.

        Parameters
        ----------
        gf : tuple of dyson.Lehmann, optional
            Green's function for each spin channel. If `None`, use
            `self.gf`. Default value is `None`.
        se : tuple of dyson.Lehmann, optional
            Self-energy for each spin channel. If `None`, use `self.se`.
            Default value is `None`.
        g0 : bool, optional
            If `True`, use the mean-field Green's function. Default
            value is `True`.

        Returns
        -------
        e_2b : float
            Two-body energy.
        """

        # Get the Green's function and self-energy
        if gf is None:
            gf = self.gf
        if se is None:
            se = self.se

        # Calculate the Galitskii--Migdal energy
        if g0:
            e_2b_α = energy.galitskii_migdal_g0(self.mo_energy[0], self.mo_occ[0], se[0])
            e_2b_β = energy.galitskii_migdal_g0(self.mo_energy[1], self.mo_occ[1], se[1])
        else:
            e_2b_α = energy.galitskii_migdal(gf[0], se[0])
            e_2b_β = energy.galitskii_migdal(gf[1], se[1])

        # Add the parts
        e_2b = (e_2b_α + e_2b_β) / 2

        return e_2b

    def init_gf(self, mo_energy=None):
        """Initialise the mean-field Green's function.

        Parameters
        ----------
        mo_energy : tuple of numpy.ndarray, optional
            Molecular orbital energies for each spin channel. Default
            value is `self.mo_energy`.

        Returns
        -------
        gf : tuple of dyson.Lehmann
            Mean-field Green's function for each spin channel.
        """

        # Get the MO energies
        if mo_energy is None:
            mo_energy = self.mo_energy

        # Build the Green's functions
        gf = [
            Lehmann(mo_energy[0], np.eye(self.nmo[0])),
            Lehmann(mo_energy[1], np.eye(self.nmo[1])),
        ]

        # Find the chemical potentials
        gf[0].chempot, _ = search_chempot(
            gf[0].energies,
            gf[0].couplings,
            self.nmo[0],
            self.nocc[0],
            occupancy=1,
        )
        gf[1].chempot, _ = search_chempot(
            gf[1].energies,
            gf[1].couplings,
            self.nmo[1],
            self.nocc[1],
            occupancy=1,
        )

        return tuple(gf)
