"""
Spin-restricted one-shot GW via self-energy moment constraints for
molecular systems.
"""

import numpy as np
from dyson import MBLSE, Lehmann, MixedMBLSE

from momentGW import energy, logging, mpi_helper, thc, util
from momentGW.base import BaseGW
from momentGW.fock import FockLoop, minimize_chempot, search_chempot
from momentGW.ints import Integrals
from momentGW.rpa import dRPA
from momentGW.tda import dTDA


def kernel(
    gw,
    nmom_max,
    moments=None,
    integrals=None,
):
    """Moment-constrained one-shot GW.

    Parameters
    ----------
    gw : BaseGW
        GW object.
    nmom_max : int
        Maximum moment number to calculate.
    moments : tuple of numpy.ndarray, optional
        Tuple of (hole, particle) moments, if passed then they will
        be used instead of calculating them. Default value is `None`.
    integrals : Integrals, optional
        Integrals object. If `None`, generate from scratch. Default
        value is `None`.

    Returns
    -------
    conv : bool
        Convergence flag. Always `True` for GW, returned for
        compatibility with other GW methods.
    gf : dyson.Lehmann
        Green's function object
    se : dyson.Lehmann
        Self-energy object
    qp_energy : numpy.ndarray
        Quasiparticle energies. Always `None` for GW, returned for
        compatibility with other GW methods.
    """

    if integrals is None:
        integrals = gw.ao2mo()

    # Get the static part of the SE
    se_static = gw.build_se_static(integrals)

    # Get the moments of the SE
    if moments is None:
        th, tp = gw.build_se_moments(
            nmom_max,
            integrals,
            mo_energy=dict(
                g=gw.mo_energy,
                w=gw.mo_energy,
            ),
        )
    else:
        th, tp = moments

    # Solve the Dyson equation
    gf, se = gw.solve_dyson(th, tp, se_static, integrals=integrals)
    conv = True

    return conv, gf, se, None


class GW(BaseGW):  # noqa: D101
    __doc__ = BaseGW.__doc__.format(
        description="Spin-restricted one-shot GW via self-energy moment constraints for molecules.",
        extra_parameters="",
    )

    @property
    def name(self):
        """Method name."""
        polarizability = self.polarizability.upper().replace("DTDA", "dTDA").replace("DRPA", "dRPA")
        return f"{polarizability}-G0W0"

    _kernel = kernel

    @logging.with_timer("Static self-energy")
    @logging.with_status("Building static self-energy")
    def build_se_static(self, integrals):
        """
        Build the static part of the self-energy, including the Fock
        matrix.

        Parameters
        ----------
        integrals : Integrals
            Integrals object.

        Returns
        -------
        se_static : numpy.ndarray
            Static part of the self-energy. If `self.diagonal_se`,
            non-diagonal elements are set to zero.
        """

        if getattr(self._scf, "xc", "hf") == "hf":
            se_static = np.zeros_like(self._scf.make_rdm1(mo_coeff=self.mo_coeff))
        else:
            with util.SilentSCF(self._scf):
                dm = self._scf.make_rdm1(mo_coeff=self.mo_coeff)
                veff = self._scf.get_veff(None, dm)
                vj = self._scf.get_j(None, dm)

            vhf = integrals.get_veff(dm, j=vj, basis="ao")
            se_static = vhf - veff
            se_static = util.einsum(
                "...pq,...pi,...qj->...ij", se_static, np.conj(self.mo_coeff), self.mo_coeff
            )

        if self.diagonal_se:
            se_static = util.einsum("...pq,pq->...pq", se_static, np.eye(se_static.shape[-1]))

        se_static += util.einsum("...p,...pq->...pq", self.mo_energy, np.eye(se_static.shape[-1]))

        return se_static

    def build_se_moments(self, nmom_max, integrals, **kwargs):
        """Build the moments of the self-energy.

        Parameters
        ----------
        nmom_max : int
            Maximum moment number to calculate.
        integrals : Integrals
            Integrals object.

        See functions in `momentGW.rpa` for `kwargs` options.

        Returns
        -------
        se_moments_hole : numpy.ndarray
            Moments of the hole self-energy. If `self.diagonal_se`,
            non-diagonal elements are set to zero.
        se_moments_part : numpy.ndarray
            Moments of the particle self-energy. If `self.diagonal_se`,
            non-diagonal elements are set to zero.
        """

        if self.polarizability.lower() == "drpa":
            rpa = dRPA(self, nmom_max, integrals, **kwargs)
            return rpa.kernel()

        elif self.polarizability.lower() == "drpa-exact":
            rpa = dRPA(self, nmom_max, integrals, **kwargs)
            return rpa.kernel(exact=True)

        elif self.polarizability.lower() == "dtda":
            tda = dTDA(self, nmom_max, integrals, **kwargs)
            return tda.kernel()

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
        integrals : Integrals
            Integrals object.
        """

        if self.polarizability.lower().startswith("thc"):
            cls = thc.Integrals
            kwargs = self.thc_opts
        else:
            cls = Integrals
            kwargs = dict(
                compression=self.compression,
                compression_tol=self.compression_tol,
                # Note: `pyscf.pbc.df` methods don't use `self.prange`
                # so the MPI solution won't work. Storing the full
                # tensor is a workaround.
                store_full=self.fock_loop or hasattr(self.with_df, "kpts"),
            )

        integrals = cls(
            self.with_df,
            self.mo_coeff,
            self.mo_occ,
            **kwargs,
        )
        if transform:
            integrals.transform()

        return integrals

    def solve_dyson(self, se_moments_hole, se_moments_part, se_static, integrals=None):
        """
        Solve the Dyson equation due to a self-energy resulting from a
        list of hole and particle moments, along with a static
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
            Moments of the hole self-energy.
        se_moments_part : numpy.ndarray
            Moments of the particle self-energy.
        se_static : numpy.ndarray
            Static part of the self-energy.
        integrals : Integrals
            Integrals object. Required if `self.fock_loop` is `True`.
            Default value is `None`.

        Returns
        -------
        gf : dyson.Lehmann
            Green's function.
        se : dyson.Lehmann
            Self-energy.
        """

        with logging.with_modifiers(status="Solving Dyson equation", timer="Dyson equation"):
            solver_occ = MBLSE(se_static, np.array(se_moments_hole))
            solver_occ.kernel()

            solver_vir = MBLSE(se_static, np.array(se_moments_part))
            solver_vir.kernel()

            solver = MixedMBLSE(solver_occ, solver_vir)
            se = solver.get_self_energy()

        if self.optimise_chempot:
            with logging.with_status("Optimising chemical potential"):
                se, opt = minimize_chempot(se, se_static, self.nocc * 2)

        error = self.moment_error(se_moments_hole, se_moments_part, se)
        logging.write(
            f"Error in moments:  [{logging.rate(sum(error), 1e-12, 1e-8)}]{sum(error):.3e}[/] "
            f"(hole = [{logging.rate(error[0], 1e-12, 1e-8)}]{error[0]:.3e}[/], "
            f"particle = [{logging.rate(error[1], 1e-12, 1e-8)}]{error[1]:.3e}[/])"
        )

        gf = Lehmann(*se.diagonalise_matrix_with_projection(se_static))
        gf.energies = mpi_helper.bcast(gf.energies, root=0)
        gf.couplings = mpi_helper.bcast(gf.couplings, root=0)
        gf.chempot = se.chempot

        if self.fock_loop:
            logging.write("")
            with logging.with_status("Running Fock loop"):
                solver = FockLoop(self, gf=gf, se=se, **self.fock_opts)
                conv, gf, se = solver.kernel(integrals=integrals)

        cpt, error = search_chempot(
            gf.energies,
            gf.couplings,
            gf.nphys,
            self.nocc * 2,
        )
        se.chempot = cpt
        gf.chempot = cpt

        logging.write("")
        style = logging.rate(
            abs(error),
            1e-6,
            1e-6 if self.fock_loop or self.optimise_chempot else 1e-1,
        )
        logging.write(f"Error in number of electrons:  [{style}]{error:.3e}[/]")
        logging.write(f"Chemical potential:  {cpt:.6f}")

        return gf, se

    def make_rdm1(self, gf=None):
        """Get the first-order reduced density matrix.

        Parameters
        ----------
        gf : dyson.Lehmann, optional
            Green's function. If `None`, use either `self.gf`, or the
            mean-field Green's function. Default value is `None`.

        Returns
        -------
        rdm1 : numpy.ndarray
            First-order reduced density matrix.
        """

        if gf is None:
            gf = self.gf
        if gf is None:
            gf = self.init_gf()

        return gf.occupied().moment(0) * 2.0

    def moment_error(self, se_moments_hole, se_moments_part, se):
        """Return the error in the moments.

        Parameters
        ----------
        se_moments_hole : numpy.ndarray
            Moments of the hole self-energy.
        se_moments_part : numpy.ndarray
            Moments of the particle self-energy.
        se : dyson.Lehmann
            Self-energy.

        Returns
        -------
        eh : float
            Error in the hole moments.
        ep : float
            Error in the particle moments.
        """

        eh = self._moment_error(
            se_moments_hole,
            se.occupied().moment(range(len(se_moments_hole))),
        )
        ep = self._moment_error(
            se_moments_part,
            se.virtual().moment(range(len(se_moments_part))),
        )

        return eh, ep

    def energy_nuc(self):
        """Calculate the nuclear repulsion energy."""
        with util.SilentSCF(self._scf):
            return self._scf.energy_nuc()

    @logging.with_timer("Energy")
    @logging.with_status("Calculating energy")
    def energy_hf(self, gf=None, integrals=None):
        """Calculate the one-body (Hartree--Fock) energy.

        Parameters
        ----------
        gf : dyson.Lehmann, optional
            Green's function. If `None`, use either `self.gf`, or the
            mean-field Green's function. Default value is `None`.
        integrals : Integrals, optional
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
                "pq,pi,qj->ij", self._scf.get_hcore(), self.mo_coeff.conj(), self.mo_coeff
            )
        rdm1 = self.make_rdm1(gf=gf)
        fock = integrals.get_fock(rdm1, h1e)

        return energy.hartree_fock(rdm1, fock, h1e)

    @logging.with_timer("Energy")
    @logging.with_status("Calculating energy")
    def energy_gm(self, gf=None, se=None, g0=True):
        r"""Calculate the two-body (Galitskii--Migdal) energy.

        Parameters
        ----------
        gf : dyson.Lehmann, optional
            Green's function. If `None`, use `self.gf`. Default value
            is `None`.
        se : dyson.Lehmann, optional
            Self-energy. If `None`, use `self.se`. Default value is
            `None`.
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
            e_2b = energy.galitskii_migdal_g0(self.mo_energy, self.mo_occ, se)
        else:
            e_2b = energy.galitskii_migdal(gf, se)

        return e_2b

    def init_gf(self, mo_energy=None):
        """Initialise the mean-field Green's function.

        Parameters
        ----------
        mo_energy : numpy.ndarray, optional
            Molecular orbital energies. Default value is
            `self.mo_energy`.

        Returns
        -------
        gf : dyson.Lehmann
            Mean-field Green's function.
        """

        if mo_energy is None:
            mo_energy = self.mo_energy

        gf = Lehmann(mo_energy, np.eye(self.nmo))
        gf.chempot = search_chempot(gf.energies, gf.couplings, self.nmo, self.nocc * 2)[0]

        return gf
