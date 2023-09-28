"""
Spin-restricted Bethe-Salpeter equation (BSE) via self-energy moment
constraints for molecular systems.
"""

import numpy as np
from dyson import CPGF, MBLGF, NullLogger
from pyscf import lib
from pyscf.lib import logger

from momentGW.base import Base
from momentGW.ints import Integrals
from momentGW.rpa import dRPA
from momentGW.tda import dTDA


def kernel(
    bse,
    nmom_max,
    mo_energy,
    mo_coeff,
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
    mo_energy : numpy.ndarray
        Molecular orbital energies.
    mo_coeff : numpy.ndarray
        Molecular orbital coefficients.
    moments : numpy.ndarray, optional
        Moments of the dynamic polarizability, if passed then they will
        be used instead of calculating them. Default value is `None`.
    integrals : Integrals, optional
        Integrals object. If `None`, generate from scratch. Default
        value is `None`.
    """

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

    # --- Extra BSE options

    excitation = "singlet"
    polarizability = None

    _opts = Base._opts + ["excitation", "polarizability"]

    def __init__(self, gw, **kwargs):
        super().__init__(gw._scf, **kwargs)

        self.gw = gw
        if self.polarizability is None:
            self.polarizability = gw.polarizability

        if self.gw.compression:
            raise NotImplementedError("Currently require `gw.compression=None` for BSE")

        # Do not modify:
        self.gf = None

        self._keys = set(self.__dict__.keys()).union(self._opts)

    @property
    def name(self):
        """Method name."""
        polarizability = self.polarizability.upper().replace("DTDA", "dTDA").replace("DRPA", "dRPA")
        return f"{polarizability}-BSE"

    _kernel = kernel

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

        integrals = Integrals(
            self.with_df,
            self.mo_coeff,
            self.mo_occ,
            compression=self.gw.compression,
            compression_tol=self.gw.compression_tol,
            store_full=True,
        )
        if transform:
            integrals.transform()

        return integrals

    def build_dd_moment_inv(self, integrals, **kwargs):
        """
        Build the first inverse moment of the density-density response.

        Parameters
        ----------
        integrals : Integrals
            Integrals object.

        See functions in `momentGW.rpa` for `kwargs` options.

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

    def build_matvec(self, integrals, moment=None):
        """
        Build the matrix-vector product required for the
        Bethe-Salpeter equation.

        Parameters
        ----------
        integrals : Integrals
            Integrals object.
        moment : numpy.ndarray, optional
            First inverse (`n=-1`) moment of the density-density
            response. If not provided, calculate from scratch. Default
            value is `None`.

        Returns
        -------
        matvec : callable
            Function that takes a vector `x` and returns the matrix-
            vector product `xA`.
        """

        # Construct the energy differences
        if not self.gw.converged:
            logger.warn(self, "GW calculation has not converged - using MO energies for BSE")
            qp_energy = self.mo_energy
        else:
            # Just use the QP energies - we could do the entire BSE in
            # the basis of the GW solution but that's more annoying
            qp_energy = self.gw.qp_energy

        d_full = lib.direct_sum(
            "a-i->ia",
            self.mo_energy[self.mo_occ == 0],
            self.mo_energy[self.mo_occ > 0],
        )
        nocc, nvir = d_full.shape

        # Get the inverse moment
        if moment is None:
            moment = self.build_dd_moment_inv(integrals)
        moment = moment.reshape(-1, nocc, nvir)

        # Get the integrals
        Lpq = integrals.Lpq
        o = slice(None, nocc)
        v = slice(nocc, None)

        # TODO does this also work with RPA-BSE?

        # Intermediates for the screened interaction to reduce the
        # number of N^4 operations in `matvec`.
        # TODO: account for this derivation!!
        q_ov = lib.einsum("Lkc,Qkc,kc->LQ", Lpq[:, o, v], Lpq[:, o, v], 1.0 / d_full)
        eta_aux = lib.einsum("Pld,Qld->PQ", Lpq[:, o, v], moment)
        q_full = q_ov - np.dot(q_ov, eta_aux)
        q_full_vv = lib.einsum("LQ,Qab->Lab", q_full, Lpq[:, v, v])
        q_full_vv_plus_vv = 4.0 * q_full_vv - Lpq[:, v, v]

        def matvec(vec):
            """
            Matrix-vector product. Input matrix should be of shape
            (aux, occ*vir).
            """

            shape = vec.shape
            vec = vec.reshape(-1, nocc, nvir)
            out = np.zeros_like(vec)

            # r_{x, ia} = v_{x, ia} (ϵ_a - ϵ_i)
            out = lib.einsum("xia,a->xia", vec, qp_energy[v])
            out -= lib.einsum("xia,i->xia", vec, qp_energy[o])

            # r_{x, jb} = v_{x, ia} κ (ia|jb)
            if self.excitation == "singlet":
                out += lib.einsum("xia,Lia,Ljb->xjb", vec, Lpq[:, o, v], Lpq[:, o, v]) * 2

            # r_{x, jb} = - v_{x, ia} (ab|ij)
            # r_{x, jb} = -2 v_{x, ia} (ij|kc) [η^{-1}]_{kc, ld} (ld|ab)
            out += lib.einsum("xia,Lij,Lab->xjb", vec, Lpq[:, o, o], q_full_vv_plus_vv)

            return out.reshape(shape)

        return matvec

    def build_dp_moments(self, nmom_max, integrals, matvec=None):
        """Build the moments of the dynamic polarizability.

        Parameters
        ----------
        nmom_max : int
            Maximum moment number to calculate.
        integrals : Integrals
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

        cput0 = (lib.logger.process_clock(), lib.logger.perf_counter())
        lib.logger.info(self, "Building dynamic polarizability moments")

        # Get the matrix-vector product callable
        if matvec is None:
            matvec = self.build_matvec(integrals)

        # Get the dipole matrices
        with self.mol.with_common_orig((0, 0, 0)):
            dip = self.mol.intor_symmetric("int1e_r", comp=3)

        # Rotate into ia basis
        ci = integrals.mo_coeff[:, integrals.mo_occ > 0]
        ca = integrals.mo_coeff[:, integrals.mo_occ == 0]
        dip = lib.einsum("xpq,pi,qa->xia", dip, ci.conj(), ca)
        dip = dip.reshape(3, -1)

        # Get the moments of the dynamic polarizability
        moments_dp = np.zeros((nmom_max + 1, 3, dip.shape[1]))
        moments_dp[0] = dip.copy()
        for n in range(1, nmom_max + 1):
            moments_dp[n] = matvec(moments_dp[n - 1])

        moments_dp = lib.einsum("px,nqx->npq", dip.conj(), moments_dp)

        lib.logger.timer(self, "moments", *cput0)

        return moments_dp

    def solve_bse(self, moments):
        """Solve the Bethe-Salpeter equation.

        Parameters
        ----------
        moments : numpy.ndarray
            Moments of the dynamic polarizability.
        """

        nlog = NullLogger()

        solver = MBLGF(np.array(moments), log=nlog)
        solver.kernel()

        gf = solver.get_greens_function()

        return gf

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
            Chebyshev moments of the dynamic polarizability, if passed
            then they will be used instead of calculating them. Default
            value is `None`.
        integrals : Integrals, optional
            Integrals object. If `None`, generate from scratch. Default
            value is `None`.
        """

        if mo_coeff is None:
            mo_coeff = self.mo_coeff
        if mo_energy is None:
            mo_energy = self.mo_energy

        cput0 = (logger.process_clock(), logger.perf_counter())
        self.dump_flags()
        logger.info(self, "nmom_max = %d", nmom_max)

        self.gf = self._kernel(
            nmom_max,
            mo_energy,
            mo_coeff,
            integrals=integrals,
            moments=moments,
        )

        for n in range(min(10, self.gf.naux)):
            en = -self.gf.energies[-(n + 1)]
            vn = self.gf.couplings[:, -(n + 1)]
            qpwt = np.linalg.norm(vn) ** 2
            logger.note(self, "EE energy level %d E = %.16g  QP weight = %0.6g", n, en, qpwt)

        logger.timer(self, self.name, *cput0)

        return self.gf


class cpBSE(BSE):
    """Chebyshev-polynomial Bethe-Salpeter equation.

    Parameters
    ----------
    mf : pyscf.scf.SCF
        PySCF mean-field class.
    scale : tuple of int
        Scaling parameters used to scale the spectrum to [-1, 1],
        given as `(a, b)` where

            a = (ωmax - ωmin) / (2 - ε)
            b = (ωmax + ωmin) / 2

        where ωmax and ωmin are the maximum and minimum energies in
        the spectrum, respectively, and ε is a small number shifting
        the spectrum values away from the boundaries.
    grid : numpy.ndarray
        Grid to plot spectral function on.
    eta : float, optional
        Regularisation parameter.  Default value is 0.1.
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

        # Do not modify:
        self.scale = kwargs.pop("scale", None)
        self.grid = kwargs.pop("grid", None)
        self.eta = kwargs.pop("eta", 0.1)

        if self.scale is None:
            raise ValueError("Must provide `scale` parameter.")
        if self.grid is None:
            raise ValueError("Must provide `grid` parameter.")

        self._keys = set(self.__dict__.keys()).union(self._opts)

    @property
    def name(self):
        """Method name."""
        polarizability = self.polarizability.upper().replace("DTDA", "dTDA").replace("DRPA", "dRPA")
        return f"{polarizability}-cpBSE"

    def build_dp_moments(self, nmom_max, integrals, matvec=None):
        """Build the moments of the dynamic polarizability.

        Parameters
        ----------
        nmom_max : int
            Maximum moment number to calculate.
        integrals : Integrals
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

        cput0 = (lib.logger.process_clock(), lib.logger.perf_counter())
        lib.logger.info(self, "Building dynamic polarizability moments")

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
        dip = lib.einsum("xpq,pi,qa->xia", dip, ci.conj(), ca)
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

        lib.logger.timer(self, "moments", *cput0)

        return moments_dp

    def solve_bse(self, moments):
        """Solve the Bethe-Salpeter equation.

        Parameters
        ----------
        moments : numpy.ndarray
            Chebyshev moments of the dynamic polarizability.
        """

        nlog = NullLogger()

        solver = CPGF(
            np.array(moments),
            self.grid,
            self.scale,
            eta=self.eta,
            # Maybe these are unnecessary?
            trace=False,
            include_real=True,
            log=nlog,
        )
        gf = solver.kernel()

        return gf

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
            Chebyshev moments of the dynamic polarizability, if passed
            then they will be used instead of calculating them. Default
            value is `None`.
        integrals : Integrals, optional
            Integrals object. If `None`, generate from scratch. Default
            value is `None`.
        """

        if mo_coeff is None:
            mo_coeff = self.mo_coeff
        if mo_energy is None:
            mo_energy = self.mo_energy

        cput0 = (logger.process_clock(), logger.perf_counter())
        self.dump_flags()
        logger.info(self, "nmom_max = %d", nmom_max)

        self.gf = self._kernel(
            nmom_max,
            mo_energy,
            mo_coeff,
            integrals=integrals,
            moments=moments,
        )

        logger.timer(self, self.name, *cput0)

        return self.gf
