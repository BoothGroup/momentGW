"""
Spin-restricted one-shot GW via self-energy moment constraints for
periodic systems.
"""

import numpy as np
from dyson import MBLSE, Lehmann, MixedMBLSE, NullLogger

from momentGW import energy, logging, mpi_helper, util
from momentGW.gw import GW
from momentGW.pbc import thc
from momentGW.pbc.base import BaseKGW
from momentGW.pbc.fock import (
    fock_loop,
    minimize_chempot,
    search_chempot,
    search_chempot_unconstrained,
)
from momentGW.pbc.ints import KIntegrals
from momentGW.pbc.rpa import dRPA
from momentGW.pbc.tda import dTDA


class KGW(BaseKGW, GW):  # noqa: D101
    __doc__ = BaseKGW.__doc__.format(
        description="Spin-restricted one-shot GW via self-energy moment constraints for "
        + "periodic systems.",
        extra_parameters="",
    )

    _opts = util.list_union(BaseKGW._opts, GW._opts)

    @property
    def name(self):
        """Define the name of the method being used."""
        polarizability = self.polarizability.upper().replace("DTDA", "dTDA").replace("DRPA", "dRPA")
        return f"{polarizability}-KG0W0"

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
        """

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

        integrals = cls(
            self.with_df,
            self.kpts,
            self.mo_coeff,
            self.mo_occ,
            **kwargs,
        )
        if transform:
            if "input_path" in kwargs and kwargs["input_path"] is not None:
                integrals.get_cderi_from_thc()
            else:
                integrals.transform()

        return integrals

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
            Density-fitted integrals.  Required if `self.fock_loop`
            is `True`.  Default value is `None`.

        Returns
        -------
        gf : list of dyson.Lehmann
            Green's function at each k-point.
        se : list of dyson.Lehmann
            Self-energy at each k-point.
        """

        with logging.with_modifiers(status="Solving Dyson equation", timer="Dyson equation"):
            se = []
            for k in self.kpts.loop(1):
                solver_occ = MBLSE(se_static[k], np.array(se_moments_hole[k]), log=NullLogger())
                solver_occ.kernel()

                solver_vir = MBLSE(se_static[k], np.array(se_moments_part[k]), log=NullLogger())
                solver_vir.kernel()

                solver = MixedMBLSE(solver_occ, solver_vir)
                se.append(solver.get_self_energy())

        if self.optimise_chempot:
            with logging.with_status("Optimising chemical potential"):
                se, opt = minimize_chempot(se, se_static, sum(self.nocc) * 2)

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

        gf = []
        for k in self.kpts.loop(1):
            g = Lehmann(*se[k].diagonalise_matrix_with_projection(se_static[k]))
            g.energies = mpi_helper.bcast(g.energies, root=0)
            g.couplings = mpi_helper.bcast(g.couplings, root=0)
            g.chempot = se[k].chempot
            gf.append(g)

        if self.fock_loop:
            logging.write("")
            with logging.with_status("Running Fock loop"):
                gf, se, conv = fock_loop(self, gf, se, integrals=integrals, **self.fock_opts)

        w = [g.energies for g in gf]
        v = [g.couplings for g in gf]
        cpt, error = search_chempot(w, v, self.nmo, sum(self.nocc) * 2)
        for g, s in zip(gf, se):
            g.chempot = cpt
            s.chempot = cpt

        logging.write("")
        style = logging.rate(
            error,
            1e-6,
            1e-6 if self.fock_loop or self.optimise_chempot else 1e-1,
        )
        logging.write(f"Error in number of electrons:  [{style}]{error:.3e}[/]")
        logging.write(f"Chemical potential (Γ):  {cpt:.6f}")

        return tuple(gf), tuple(se)

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

        if gf is None:
            gf = self.gf
        if gf is None:
            gf = self.init_gf()

        return np.array([g.occupied().moment(0) for g in gf]) * 2

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

        if gf is None:
            gf = self.gf
        if integrals is None:
            integrals = self.ao2mo()

        with util.SilentSCF(self._scf):
            h1e = util.einsum(
                "kpq,kpi,kqj->kij", self._scf.get_hcore(), self.mo_coeff.conj(), self.mo_coeff
            )
        rdm1 = self.make_rdm1()
        fock = integrals.get_fock(rdm1, h1e)

        e_1b = sum(energy.hartree_fock(rdm1[k], fock[k], h1e[k]) for k in self.kpts.loop(1))
        e_1b /= self.nkpts

        return e_1b.real

    def energy_gm(self, gf=None, se=None, g0=True):
        r"""Calculate the two-body (Galitskii--Migdal) energy.

        Parameters
        ----------
        gf : tuple of dyson.Lehmann, optional
            Green's function at each k-point. If `None`, use `self.gf`.
            Default value is `None`.
        se : tuple of dyson.Lehmann, optional
            Self-energy at each k-point. If `None`, use `self.se`.
            Default value is `None`.
        g0 : bool, optional
            If `True`, use the mean-field Green's function. Default
            value is `True`.

        Returns
        -------
        e_2b : float
            Two-body energy.

        Notes
        -----
        With `g0=False`, this function scales as
        :math:`\mathcal{O}(N^4)` with system size, whereas with
        `g0=True`, it scales as :math:`\mathcal{O}(N^3)`.
        """

        if gf is None:
            gf = self.gf
        if se is None:
            se = self.se

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
