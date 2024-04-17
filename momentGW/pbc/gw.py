"""
Spin-restricted one-shot GW via self-energy moment constraints for
periodic systems.
"""

import numpy as np
from dyson import MBLSE, Lehmann, MixedMBLSE

from momentGW import energy, logging, util
from momentGW import GW
from momentGW.pbc import thc
from momentGW.pbc.base import BaseKGW
from momentGW.pbc.fock import FockLoop, search_chempot_unconstrained
from momentGW.pbc.ints import KIntegrals
from momentGW.pbc.tda import dTDA
from momentGW.pbc.rpa import dRPA


class KGW(BaseKGW, GW):
    """
    Spin-restricted one-shot GW via self-energy moment constraints for
    periodic systems.

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

    _defaults = util.dict_union(BaseKGW._defaults, GW._defaults)

    @property
    def name(self):
        """Get the method name."""
        polarizability = self.polarizability.upper().replace("DTDA", "dTDA").replace("DRPA", "dRPA")
        return f"{polarizability}-KG0W0"

    @logging.with_timer("Static self-energy")
    @logging.with_status("Building static self-energy")
    def build_se_static(self, integrals):
        """
        Build the static part of the self-energy, including the Fock
        matrix.

        Parameters
        ----------
        integrals : KIntegrals
            Integrals object.

        Returns
        -------
        se_static : numpy.ndarray
            Static part of the self-energy at each k-point. If
            `self.diagonal_se`, non-diagonal elements are set to zero.
        """
        return super().build_se_static(integrals)

    def build_se_moments(self, nmom_max, integrals, **kwargs):
        """Build the moments of the self-energy.

        Parameters
        ----------
        nmom_max : int
            Maximum moment number to calculate.
        integrals : KIntegrals
            Density-fitted integrals.

        See functions in `momentGW.rpa` for `kwargs` options.

        Returns
        -------
        se_moments_hole : numpy.ndarray
            Moments of the hole self-energy at each k-point. If
            `self.diagonal_se`, non-diagonal elements are set to zero.
        se_moments_part : numpy.ndarray
            Moments of the particle self-energy at each k-point. If
            `self.diagonal_se`, non-diagonal elements are set to zero.
        """

        if self.polarizability.lower() == "dtda":
            tda = dTDA(self, nmom_max, integrals, **kwargs)
            return tda.kernel()
        if self.polarizability.lower() == "drpa":
            rpa = dRPA(self, nmom_max, integrals, **kwargs)
            return rpa.kernel()
        elif self.polarizability.lower() == "thc-dtda":
            tda = thc.dTDA(self, nmom_max, integrals, **kwargs)
            return tda.kernel()
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
        integrals : KIntegrals
            Integrals object.

        See Also
        --------
        momentGW.pbc.ints.KIntegrals
        momentGW.pbc.thc.KIntegrals
        """

        # Get the integrals class
        if self.polarizability.lower().startswith("thc"):
            cls = thc.KIntegrals
            kwargs = self.thc_opts
        else:
            cls = KIntegrals
            kwargs = dict(
                compression=self.compression,
                compression_tol=self.compression_tol,
                store_full=self.fock_loop,
                input_path=self.thc_opts["file_path"],
            )

        # Get the integrals
        integrals = cls(
            self.with_df,
            self.kpts,
            self.mo_coeff,
            self.mo_occ,
            **kwargs,
        )

        # Transform the integrals
        if transform:
            if "input_path" in kwargs and kwargs["input_path"] is not None:
                integrals.get_cderi_from_thc()
            else:
                integrals.transform()

        return integrals

    def solve_dyson(self, se_moments_hole, se_moments_part, se_static, integrals=None):
        """Solve the Dyson equation due to a self-energy resulting
        from a list of hole and particle moments, along with a static
        contribution.

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
        se_moments_hole : numpy.ndarray
            Moments of the hole self-energy at each k-point.
        se_moments_part : numpy.ndarray
            Moments of the particle self-energy at each k-point.
        se_static : numpy.ndarray
            Static part of the self-energy at each k-point.
        integrals : KIntegrals, optional
            Density-fitted integrals. Required if `self.fock_loop`
            is `True`. Default value is `None`.

        Returns
        -------
        gf : tuple of dyson.Lehmann
            Green's function at each k-point.
        se : tuple of dyson.Lehmann
            Self-energy at each k-point.

        See Also
        --------
        momentGW.pbc.fock.FockLoop
        """

        # Solve the Dyson equation for the moments
        with logging.with_modifiers(status="Solving Dyson equation", timer="Dyson equation"):
            se = []
            for k in self.kpts.loop(1):
                solver_occ = MBLSE(se_static[k], np.array(se_moments_hole[k]))
                solver_occ.kernel()

                solver_vir = MBLSE(se_static[k], np.array(se_moments_part[k]))
                solver_vir.kernel()

                solver = MixedMBLSE(solver_occ, solver_vir)
                se.append(solver.get_self_energy())

        # Initialise the solver
        solver = FockLoop(self, se=se, **self.fock_opts)

        # Shift the self-energy poles relative to the Green's function
        # to better conserve the particle number
        if self.optimise_chempot:
            se = solver.auxiliary_shift(se_static)

        # Find the error in the moments
        error_h, error_p = zip(
            *(
                self.moment_error(th, tp, s)
                for th, tp, s in zip(se_moments_hole, se_moments_part, se)
            )
        )
        error = (sum(error_h), sum(error_p))
        logging.write(
            f"Error in moments:  [{logging.rate(sum(error), 1e-12, 1e-8)}]{sum(error):.3e}[/] "
            f"(hole = [{logging.rate(error[0], 1e-12, 1e-8)}]{error[0]:.3e}[/], "
            f"particle = [{logging.rate(error[1], 1e-12, 1e-8)}]{error[1]:.3e}[/])"
        )

        # Solve the Dyson equation for the self-energy
        gf, error = solver.solve_dyson(se_static)
        for g, s in zip(gf, se):
            s.chempot = g.chempot

        # Self-consistently renormalise the density matrix
        if self.fock_loop:
            logging.write("")
            solver.gf = gf
            solver.se = se
            conv, gf, se = solver.kernel(integrals=integrals)
            _, error = solver.search_chempot(gf)

        # Print the error in the number of electrons
        logging.write("")
        style = logging.rate(
            error,
            1e-6,
            1e-6 if self.fock_loop or self.optimise_chempot else 1e-1,
        )
        logging.write(f"Error in number of electrons:  [{style}]{error:.3e}[/]")
        logging.write(f"Chemical potential (Γ):  {gf[0].chempot:.6f}")

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
            Tuple of (hole, particle) moments at each k-point, if passed
            then they will be used instead of calculating them. Default
            value is `None`.
        integrals : KIntegrals, optional
            Integrals object. If `None`, generate from scratch. Default
            value is `None`.

        Returns
        -------
        converged : bool
            Whether the solver converged. For single-shot calculations,
            this is always `True`.
        gf : tuple of dyson.Lehmann
            Green's function object at each k-point.
        se : tuple of dyson.Lehmann
            Self-energy object at each k-point.
        qp_energy : NoneType
            Quasiparticle energies. For most GW methods, this is `None`.
        """
        return super().kernel(nmom_max, moments=moments, integrals=integrals)

    def make_rdm1(self, gf=None):
        """Get the first-order reduced density matrix.

        Parameters
        ----------
        gf : tuple of dyson.Lehmann, optional
            Green's function at each k-point. If `None`, use either
            `self.gf`, or the mean-field Green's function. Default
            value is `None`.

        Returns
        -------
        rdm1 : numpy.ndarray
            First-order reduced density matrix at each k-point.
        """

        # Get the Green's function
        if gf is None:
            gf = self.gf
        if gf is None:
            gf = self.init_gf()

        return np.array([g.occupied().moment(0) for g in gf]) * 2

    @logging.with_timer("Energy")
    @logging.with_status("Calculating energy")
    def energy_hf(self, gf=None, integrals=None):
        """Calculate the one-body (Hartree--Fock) energy.

        Parameters
        ----------
        gf : tuple of dyson.Lehmann, optional
            Green's function at each k-point. If `None`, use either
            `self.gf`, or the mean-field Green's function. Default
            value is `None`.
        integrals : KIntegrals, optional
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

        # Find the Fock matrix
        with util.SilentSCF(self._scf):
            h1e = util.einsum(
                "kpq,kpi,kqj->kij", self._scf.get_hcore(), self.mo_coeff.conj(), self.mo_coeff
            )
        rdm1 = self.make_rdm1()
        fock = integrals.get_fock(rdm1, h1e)

        # Calculate the Hartree--Fock energy at each k-point
        e_1b = sum(energy.hartree_fock(rdm1[k], fock[k], h1e[k]) for k in self.kpts.loop(1))
        e_1b /= self.nkpts

        return e_1b.real

    @logging.with_timer("Energy")
    @logging.with_status("Calculating energy")
    def energy_gm(self, gf=None, se=None, g0=True):
        r"""Calculate the two-body (Galitskii--Migdal) energy.

        Parameters
        ----------
        gf : tuple of tuple of dyson.Lehmann, optional
            Green's function at each k-point for each spin channel. If
            `None`, use `self.gf`. Default value is `None`.
        se : tuple of tuple of dyson.Lehmann, optional
            Self-energy at each k-point for each spin channel. If
            `None`, use `self.se`. Default value is `None`.
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
            e_2b = sum(
                energy.galitskii_migdal_g0(self.mo_energy[k], self.mo_occ[k], se[k])
                for k in self.kpts.loop(1)
            )
        else:
            e_2b = sum(energy.galitskii_migdal(gf[k], se[k]) for k in self.kpts.loop(1))

        return e_2b.real

    @logging.with_timer("Interpolation")
    @logging.with_status("Interpolating in k-space")
    def interpolate(self, mf, nmom_max):
        """
        Interpolate the object to a new k-point grid, represented by a
        new mean-field object.

        Parameters
        ----------
        mf : pyscf.pbc.scf.KSCF
            Mean-field object on new k-point mesh.
        nmom_max : int
            Maximum moment number to calculate.

        Returns
        -------
        other : __class__
            Interpolated object.
        """

        if len(mf.kpts) % len(self.kpts) != 0:
            raise ValueError("Size of interpolation grid must be a multiple of the old grid.")

        other = self.__class__(mf)
        other.__dict__.update({key: getattr(self, key) for key in self._opts})
        sc = util.einsum("kpq,kqi->kpi", mf.get_ovlp(), mf.mo_coeff)

        def interp(m):
            # Interpolate part of the moments via the AO basis
            m = util.einsum("knij,kpi,kqj->knpq", m, self.mo_coeff, self.mo_coeff.conj())
            m = np.stack(
                [self.kpts.interpolate(other.kpts, m[:, n]) for n in range(nmom_max + 1)],
                axis=1,
            )
            m = util.einsum("knpq,kpi,kqj->knij", m, sc.conj(), sc)
            return m

        # Get the moments of the self-energy on the small k-point grid
        th = np.array([se.occupied().moment(range(nmom_max + 1)) for se in self.se])
        tp = np.array([se.virtual().moment(range(nmom_max + 1)) for se in self.se])

        # Interpolate the moments
        th = interp(th)
        tp = interp(tp)

        # Get the static self-energy on the fine k-point grid
        integrals = other.ao2mo(transform=False)
        se_static = other.build_se_static(integrals)

        # Solve the Dyson equation on the fine k-point grid
        gf, se = other.solve_dyson(th, tp, se_static, integrals=integrals)

        # Set attributes
        # TODO handle _qp_energy if not None
        other.gf = gf
        other.se = se

        return other

    def init_gf(self, mo_energy=None):
        """Initialise the mean-field Green's function.

        Parameters
        ----------
        mo_energy : numpy.ndarray, optional
            Molecular orbital energies at each k-point. Default value is
            `self.mo_energy`.

        Returns
        -------
        gf : tuple of dyson.Lehmann
            Mean-field Green's function at each k-point.
        """

        if mo_energy is None:
            mo_energy = self.mo_energy

        gf = []
        for k in self.kpts.loop(1):
            gf.append(Lehmann(mo_energy[k], np.eye(self.nmo)))

        ws = [g.energies for g in gf]
        vs = [g.couplings for g in gf]
        nelec = [n * 2 for n in self.nocc]
        chempot = search_chempot_unconstrained(ws, vs, self.nmo, sum(nelec))[0]

        for k in self.kpts.loop(1):
            gf[k].chempot = chempot

        return tuple(gf)
