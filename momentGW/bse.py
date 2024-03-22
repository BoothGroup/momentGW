"""
Spin-restricted Bethe-Salpeter equation (BSE) via self-energy moment
constraints for molecular systems.
"""

import numpy as np
from dyson import CPGF, MBLGF

from momentGW import logging, mpi_helper, util
from momentGW.base import Base
from momentGW.ints import Integrals
from momentGW.rpa import dRPA
from momentGW.tda import dTDA


def kernel(
    bse,
    nmom_max,
    moments=None,
    integrals=None,
):
    """Bethe-Salpeter equation.

    Parameters
    ----------
    gw : BaseGW
        GW object.
    nmom_max : int
        Maximum moment number to calculate.
    moments : numpy.ndarray, optional
        Moments of the dynamic polarizability, if passed then they will
        be used instead of calculating them. Default value is `None`.
    integrals : BaseIntegrals, optional
        Integrals object. If `None`, generate from scratch. Default
        value is `None`.

    Returns
    -------
    gf : dyson.Lehmann
        Green's function object.
    """

    # Get the integrals
    if integrals is None:
        integrals = bse.ao2mo()

    # Get the moments of the dynamic polarizability
    if moments is None:
        moments = bse.build_dp_moments(nmom_max, integrals)

    # Solve the Bethe-Salpeter equation
    gf = bse.solve_bse(moments)

    return gf


class BSE(Base):
    """Bethe-Salpeter equation.

    Parameters
    ----------
    gw : BaseGW
        GW object.
    polarizability : str
        Type of polarizability to use, can be one of `("drpa",
        "drpa-exact", "dtda", "thc-dtda"). Default value is the same
        as the underling GW object.
    excitation : str
        Type of excitation, can be one of `("singlet", "triplet")`.
        Default value is `"singlet"`.
    """

    # --- Default BSE options

    excitation = "singlet"
    polarizability = None

    _opts = Base._opts + ["excitation", "polarizability"]

    _kernel = kernel

    def __init__(self, gw, **kwargs):
        if kwargs.get("polariability") is None:
            kwargs["polarizability"] = gw.polarizability
        super().__init__(gw._scf, **kwargs)

        # Parameters
        self.gw = gw

        # Attributes
        self.gf = None

    @property
    def name(self):
        """Get the method name."""
        polarizability = self.polarizability.upper().replace("DTDA", "dTDA").replace("DRPA", "dRPA")
        return f"{polarizability}-BSE"

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
        integrals : BaseIntegrals
            Integrals object.
        """

        # Get the integrals
        integrals = Integrals(
            self.with_df,
            self.mo_coeff,
            self.mo_occ,
            compression=self.gw.compression,
            compression_tol=self.gw.compression_tol,
            store_full=True,
        )

        # Check compression
        compression = integrals._parse_compression()
        if compression and compression != {"oo", "vv", "ov"}:
            logging.warn(
                "[bad]Running BSE with compression without including all integral blocks "
                "is not recommended[/]. See example 17.",
            )

        # Transform the integrals
        if transform:
            integrals.transform()

        return integrals

    def build_dd_moment_inv(self, integrals, **kwargs):
        """
        Build the first inverse moment of the density-density response.

        Parameters
        ----------
        integrals : BaseIntegrals
            Integrals object.
        **kwargs : dict, optional
            Additional keyword arguments to pass to the RPA or TDA
            solver. See `momentGW.tda` and `momentGW.rpa` for options.

        Returns
        -------
        moment : numpy.ndarray
            First inverse (`n=-1`) moment of the density-density
            response.
        """

        if self.polarizability == "drpa":
            rpa = dRPA(self.gw, 1, integrals, **kwargs)
            return rpa.build_dd_moment_inv()

        elif self.polarizability == "dtda":
            tda = dTDA(self.gw, 1, integrals, **kwargs)
            return tda.build_dd_moment_inv()

        else:
            raise NotImplementedError

    @logging.with_timer("Matrix-vector product construction")
    @logging.with_status("Constructing matrix-vector product")
    def build_matvec(self, integrals, moment=None):
        """
        Build the matrix-vector product required for the
        Bethe-Salpeter equation.

        Parameters
        ----------
        integrals : BaseIntegrals
            Integrals object.
        moment : numpy.ndarray, optional
            First inverse (`n=-1`) moment of the density-density
            response. If not provided, calculate from scratch. Default
            value is `None`.

        Returns
        -------
        matvec : callable
            Function that takes a vector ``x`` and returns the matrix-
            vector product ``xA``.
        """

        # Developer note: this is not parallelised much, I just made
        # sure that it would run through in parallel.

        # Construct the energy differences
        if not self.gw.converged:
            logging.warn("[red]GW calculation has not converged[/] - using MO energies for BSE")
            qp_energy = self.mo_energy
        else:
            # Just use the QP energies - we could do the entire BSE in
            # the basis of the GW solution but that's more annoying
            qp_energy = self.gw.qp_energy

        # Get the 1h1p energies
        d = util.build_1h1p_energies(self.mo_energy, self.mo_occ)
        nocc, nvir = d.shape

        # Get the inverse moment
        if moment is None:
            moment = self.build_dd_moment_inv(integrals)

        # Get the integrals
        p0, p1 = list(mpi_helper.prange(0, integrals.nmo, integrals.nmo))[0]
        Lpq = np.zeros((integrals.naux, integrals.nmo, integrals.nmo))
        Lpq_part = integrals.Lpq
        if integrals._rot is not None:
            Lpq_part = util.einsum("PQ,Pij->Qij", integrals._rot, Lpq_part)
        Lpq[:, :, p0:p1] = Lpq_part
        Lpq = mpi_helper.allreduce(Lpq)
        Loo = Lpq[:, :nocc, :nocc]
        Lvv = Lpq[:, nocc:, nocc:]
        Lov = Lpq[:, :nocc, nocc:]

        # Intermediates for the screened interaction to reduce the
        # number of N^4 operations in `matvec`.
        # TODO: account for this derivation!!
        # TODO does this also work with RPA-BSE?
        q_ov = util.einsum("Lia,Qia,ia->LQ", Lov, Lov, 1.0 / d)
        eta_aux = util.einsum("Px,Qx->PQ", integrals.Lia, moment)
        eta_aux = mpi_helper.allreduce(eta_aux)
        q_full = q_ov - np.dot(q_ov, eta_aux)
        q_full = 4.0 * q_full - np.eye(q_full.shape[0])
        q_full_vv = util.einsum("LQ,Qab->Lab", q_full, Lvv)

        @logging.with_timer("Matrix-vector product")
        @logging.with_status("Evaluating matrix-vector product")
        def matvec(vec):
            """
            Matrix-vector product. Input matrix should be of shape
            (aux, occ*vir).
            """

            shape = vec.shape
            vec = vec.reshape(-1, nocc, nvir)
            out = np.zeros_like(vec)

            # r_{x, ia} = v_{x, ia} (ϵ_a - ϵ_i)
            out = util.einsum("xia,a->xia", vec, qp_energy[nocc:])
            out -= util.einsum("xia,i->xia", vec, qp_energy[:nocc])

            # r_{x, jb} = v_{x, ia} κ (ia|jb)
            if self.excitation == "singlet":
                out += util.einsum("xia,Lia,Ljb->xjb", vec, Lov, Lov) * 2

            # r_{x, jb} = - v_{x, ia} (ab|ij)
            # r_{x, jb} = -2 v_{x, ia} (ij|kc) [η^{-1}]_{kc, ld} (ld|ab)
            # Loop over x avoids possibly big intermediates
            for x in range(vec.shape[0]):
                out[x] += util.einsum("ia,Lij,Lab->jb", vec[x], Loo, q_full_vv)

            return out.reshape(shape)

        return matvec

    @logging.with_timer("Dynamic polarizability moments")
    @logging.with_status("Constructing dynamic polarizability moments")
    def build_dp_moments(self, nmom_max, integrals, matvec=None):
        """Build the moments of the dynamic polarizability.

        Parameters
        ----------
        nmom_max : int
            Maximum moment number to calculate.
        integrals : BaseIntegrals
            Integrals object.
        matvec : callable, optional
            Function that computes the matrix-vector product between
            the Bethe-Salpeter Hamiltonian and a vector. If not
            provided, calculate using `build_matvec`. Default value is
            `None`.

        Returns
        -------
        moments_dp : numpy.ndarray
            Moments of the dynamic polarizability.
        orth : numpy.ndarray
            Orthogonalization matrix. For compatibility with the
            Chebyshev solver, and is `None` in this case.
        """

        # Get the matrix-vector product callable
        if matvec is None:
            matvec = self.build_matvec(integrals)

        # Get the dipole matrices
        with self.mol.with_common_orig((0, 0, 0)):
            dip = self.mol.intor_symmetric("int1e_r", comp=3)

        # Rotate into ia basis
        ci = integrals.mo_coeff[:, integrals.mo_occ > 0]
        ca = integrals.mo_coeff[:, integrals.mo_occ == 0]
        dip = util.einsum("xpq,pi,qa->xia", dip, ci.conj(), ca)
        dip = dip.reshape(3, -1)

        # Get the moments of the dynamic polarizability
        moments_dp = np.zeros((nmom_max + 1, 3, dip.shape[1]))
        moments_dp[0] = dip.copy()
        for n in range(1, nmom_max + 1):
            moments_dp[n] = matvec(moments_dp[n - 1])

        # Rotate basis
        moments_dp = util.einsum("px,nqx->npq", dip.conj(), moments_dp)

        return moments_dp

    def solve_bse(self, moments):
        """Solve the Bethe-Salpeter equation.

        Parameters
        ----------
        moments : numpy.ndarray
            Moments of the dynamic polarizability.

        Returns
        -------
        gf : dyson.Lehmann
            Green's function object.
        """

        solver = MBLGF(np.array(moments))
        solver.kernel()

        gf = solver.get_greens_function()

        return gf

    def _get_excitations_table(self):
        """Return the excitations as a table.

        Returns
        -------
        table : rich.Table
            Table with the excitations.
        """

        # TODO check nomenclature

        # Build table
        table = logging.Table(title="Optical excitation energies")
        table.add_column("Excitation", justify="right")
        table.add_column("Energy", justify="right", style="output")
        table.add_column("Dipole", justify="right")
        table.add_column("X", justify="right")
        table.add_column("Y", justify="right")
        table.add_column("Z", justify="right")

        # Add EEs
        for n in range(min(5, self.gf.naux)):
            en = self.gf.energies[n]
            vn = self.gf.couplings[:, n]
            weight = np.sum(vn**2)
            table.add_row(
                f"EE {n:>2}",
                f"{en:.10f}",
                f"{weight:.5f}",
                f"{vn[0]:.5f}",
                f"{vn[1]:.5f}",
                f"{vn[2]:.5f}",
            )

        return table

    def _get_summary_panel(self, timer):
        """Return the summary as a panel.

        Parameters
        ----------
        timer : Timer
            Timer object.

        Returns
        -------
        panel : rich.Panel
            Panel with the summary.
        """

        # Get the summary message
        msg = f"{self.name} ran in {timer.format_time(timer.total())}."

        # Build the table
        table = logging._Table.grid()
        table.add_row(msg)
        table.add_row("")
        table.add_row(self._get_excitations_table())

        # Build the panel
        panel = logging.Panel(table, title="Summary", padding=(1, 2), expand=False)

        return panel

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
            Chebyshev moments of the dynamic polarizability, if passed
            then they will be used instead of calculating them. Default
            value is `None`.
        integrals : BaseIntegrals, optional
            Integrals object. If `None`, generate from scratch. Default
            value is `None`.

        Returns
        -------
        gf : dyson.Lehmann
            Green's function object.
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
        logging.write("")
        with logging.with_status(f"Running {self.name} kernel"):
            self.gf = self._kernel(
                nmom_max,
                integrals=integrals,
                moments=moments,
            )
        logging.write("", comment=f"End of {self.name} kernel")

        # Print the summary in a panel
        logging.write(self._get_summary_panel(timer))

        return self.gf


class cpBSE(BSE):
    r"""Chebyshev-polynomial Bethe-Salpeter equation.

    Parameters
    ----------
    mf : pyscf.scf.SCF
        PySCF mean-field class.
    scale : tuple of int
        Scaling parameters used to scale the spectrum to ``[-1, 1]``,
        given as `(a, b)` where

        .. math::
            a = \\frac{\omega_{\max} - \omega_{\min}}{2 - \epsilon},
            b = \\frac{\omega_{\max} + \omega_{\min}}{2}.

        where :math:`\omega_{\max}` and :math:`\omega_{\min}` are the
        maximum and minimum energies in the spectrum, respectively, and
        :math:`\epsilon` is a small number shifting the spectrum values
        away from the boundaries.
    grid : numpy.ndarray
        Grid to plot spectral function on.
    eta : float, optional
        Regularisation parameter. Default value is `0.1`.
    polarizability : str, optional
        Type of polarizability to use, can be one of `("drpa",
        "drpa-exact", "dtda", "thc-dtda"). Default value is `"drpa"`.
    excitation : str, optional
        Type of excitation, can be one of `("singlet", "triplet")`.
        Default value is `"singlet"`.
    """

    # --- Extra cpBSE options

    scale = None
    grid = None
    eta = 0.1

    _opts = BSE._opts + ["scale", "grid", "eta"]

    def __init__(self, gw, **kwargs):
        super().__init__(gw, **kwargs)

        # Check options
        if self.scale is None:
            raise ValueError("Must provide `scale` parameter.")
        if self.grid is None:
            raise ValueError("Must provide `grid` parameter.")

    @property
    def name(self):
        """Get the method name."""
        polarizability = self.polarizability.upper().replace("DTDA", "dTDA").replace("DRPA", "dRPA")
        return f"{polarizability}-cpBSE"

    @logging.with_timer("Dynamic polarizability moments")
    @logging.with_status("Constructing dynamic polarizability moments")
    def build_dp_moments(self, nmom_max, integrals, matvec=None):
        """Build the moments of the dynamic polarizability.

        Parameters
        ----------
        nmom_max : int
            Maximum moment number to calculate.
        integrals : BaseIntegrals
            Integrals object.
        matvec : callable, optional
            Function that computes the matrix-vector product between
            the Bethe-Salpeter Hamiltonian and a vector. If not
            provided, calculate using `build_matvec`. Default value is
            `None`.

        Returns
        -------
        moments_dp : numpy.ndarray
            Chebyshev moments of the dynamic polarizability.
        """

        # Get the matrix-vector product callable
        if matvec is None:
            matvec = self.build_matvec(integrals)

        # Scale the matrix-vector product
        a, b = self.scale
        matvec_scaled = lambda v: matvec(v) / a - b * v / a

        # Get the dipole matrices
        with self.mol.with_common_orig((0, 0, 0)):
            dip = self.mol.intor_symmetric("int1e_r", comp=3)

        # Rotate into ia basis
        ci = integrals.mo_coeff[:, integrals.mo_occ > 0]
        ca = integrals.mo_coeff[:, integrals.mo_occ == 0]
        dip = util.einsum("xpq,pi,qa->xia", dip, ci.conj(), ca)
        dip = dip.reshape(3, -1)

        # Get the moments of the dynamic polarizability
        moments_dp = np.zeros((nmom_max + 1, 3, 3))
        vecs = (dip, matvec_scaled(dip))
        moments_dp[0] = np.dot(vecs[0], dip.T)
        moments_dp[1] = np.dot(vecs[1], dip.T)
        for i in range(2, nmom_max + 1):
            vec_next = 2.0 * matvec_scaled(vecs[1]) - vecs[0]
            moments_dp[i] = np.dot(vec_next, dip.T)
            vecs = (vecs[1], vec_next)

        return moments_dp

    def solve_bse(self, moments):
        """Solve the Bethe-Salpeter equation.

        Parameters
        ----------
        moments : numpy.ndarray
            Chebyshev moments of the dynamic polarizability.

        Returns
        -------
        gf : numpy.ndarray
            Green's function object.
        """

        solver = CPGF(
            np.array(moments),
            self.grid,
            self.scale,
            eta=self.eta,
            # Maybe these are unnecessary?
            trace=False,
            include_real=True,
        )
        gf = solver.kernel()

        return gf

    def _get_summary_panel(self, timer):
        """Return the summary as a panel.

        Parameters
        ----------
        timer : Timer
            Timer object.

        Returns
        -------
        panel : rich.Panel
            Panel with the summary.
        """

        # Get the summary message
        msg = f"{self.name} ran in {timer.format_time(timer.total())}."

        # Build the table
        table = logging._Table.grid()
        table.add_row(msg)

        # Build the panel
        panel = logging.Panel(table, title="Summary", padding=(1, 2), expand=False)

        return panel
