"""Construct TDA moments."""

import numpy as np
import scipy.special
from pyscf import lib

from momentGW import logging, mpi_helper, util
from momentGW.base import BaseSE


class dTDA(BaseSE):
    """Compute the self-energy moments using dTDA.

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
        super().__init__(gw, nmom_max, integrals)

        # Get the MO energies for G and W
        if mo_energy is not None:
            self.mo_energy_g = mo_energy["g"]
            self.mo_energy_w = mo_energy["w"]
        else:
            self.mo_energy_g = self.mo_energy_w = gw.mo_energy

        # Get the MO occupancies for G and W
        if mo_occ is not None:
            self.mo_occ_g = mo_occ["g"]
            self.mo_occ_w = mo_occ["w"]
        else:
            self.mo_occ_g = self.mo_occ_w = gw.mo_occ

        # Options and thresholds
        self.report_quadrature_error = True
        if self.gw.compression and "ia" in self.gw.compression.split(","):
            self.compression_tol = gw.compression_tol
        else:
            self.compression_tol = None

        self.d = None

    def _build_d(self):
        """Build the orbital energy differences matrix."""
        p0, p1 = self.mpi_slice(self.nov)
        self.d = util.build_1h1p_energies(self.mo_energy_w, self.mo_occ_w).ravel()[p0:p1]

    @logging.with_timer("Zeroth density-density moments")
    @logging.with_status("Constructing zeroth density-density moment")
    def build_zeroth_moment(self, m0=None):
        """Build the zeroth moment of the density-density response.

        Parameters
        ----------
        m0 : numpy.ndarray, optional
            The zeroth moment of the density-density response. If
            `None`, use `self.integrals.Lia`. This argument allows for
            custom starting points in the recursion i.e. in optical
            spectra calculations. Default value is `None`.

        Returns
        -------
        zeroth moment : numpy.ndarray
            Zeroth moment of the density-density response.
        """
        return m0 if m0 is not None else self.integrals.Lia

    @logging.with_timer("Nth density-density moments")
    @logging.with_status("Constructing nth density-density moment")
    def build_nth_dd_moment(self, n, recursion_term=None, zeroth_mom=None):
        """Build the nth moment of the density-density response.

        Parameters
        ----------
        n : int
            Moment order to be built.
        recursion_term : numpy.ndarray, optional
            Previous recursion term required to build the next moment. In the case of TDA this is
            the previous density-density response.
        zeroth moment : numpy.ndarray, optional
            Zeroth moment of the density-density response.

        Returns
        -------
        recursion_term : numpy.ndarray
            Term required for the next moment. In the case of TDA this is the current
            density-density response moment.
        eta_aux : numpy.ndarray
            The nth density-density response moment in (N_aux,N_aux) form
        """
        if recursion_term is None:
            if n != 0:
                raise AttributeError(
                    f"To build the {n}th dd-moment, a recursion_term must be provided"
                )
            if zeroth_mom is None:
                recursion_term = self.build_zeroth_moment()
            else:
                recursion_term = zeroth_mom
        else:
            tmp = np.dot(recursion_term, self.integrals.Lia.T)  # aux^2 o v
            tmp = mpi_helper.allreduce(tmp)
            recursion_term = recursion_term * self.d[None]
            recursion_term += np.dot(tmp, self.integrals.Lia) * 2.0
            del tmp
        return recursion_term, np.dot(recursion_term, self.integrals.Lia.T)  # aux^2 o v

    @logging.with_timer("Density-density moments")
    @logging.with_status("Constructing density-density moments")
    def build_dd_moments(self, m0=None):
        """Build the moments of the density-density response.

        Parameters
        ----------
        m0 : numpy.ndarray, optional
            The zeroth moment of the density-density response. If
            `None`, use `self.integrals.Lia`. This argument allows for
            custom starting points in the recursion i.e. in optical
            spectra calculations. Default value is `None`.

        Returns
        -------
        moments : numpy.ndarray
            Moments of the density-density response.
        """
        if self.d is None:
            self._build_d()

        # Initialise the moments
        naux = self.naux if m0 is None else m0.shape[0]
        p0, p1 = self.mpi_slice(self.nov)
        moments = np.zeros((self.nmom_max + 1, naux, p1 - p0))

        # Get the zeroth order moment
        moments[0] = self.build_zeroth_moment(m0=m0)

        # Get the higher order moments
        for i in range(1, self.nmom_max + 1):
            moments[i] = moments[i - 1] * self.d[None]
            tmp = np.dot(moments[i - 1], self.integrals.Lia.T)
            tmp = mpi_helper.allreduce(tmp) * 2.0
            moments[i] += np.dot(tmp, self.integrals.Lia)
            del tmp

        return moments

    def kernel(self, exact=False):
        """Run the polarizability calculation to compute moments of the self-energy.

        Parameters
        ----------
        exact : bool, optional
            Has no effect and is only present for compatibility with
            `dRPA`. Default value is `False`.

        Returns
        -------
        moments_occ : numpy.ndarray
            Moments of the occupied self-energy.
        moments_vir : numpy.ndarray
            Moments of the virtual self-energy.
        """

        # Build the self-energy moments
        moments_occ, moments_vir = self.build_se_moments()
        return moments_occ, moments_vir

    @logging.with_timer("Moment convolution")
    @logging.with_status("Convoluting moments")
    def convolve(self, eta, eta_orders=None, mo_energy_g=None, mo_occ_g=None):
        """Handle the convolution of the moments of the Green's function and screened Coulomb
        interaction.

        Parameters
        ----------
        eta : numpy.ndarray
            Moments of the density-density response partly transformed
            into moments of the screened Coulomb interaction.
        mo_energy_g : numpy.ndarray, optional
            Energies of the Green's function. If `None`, use
            `self.mo_energy_g`. Default value is `None`.
        eta_orders : list, optional
            List of orders for the rotated density-density moments in
            `eta`. If `None`, assume it spans all required orders.
            Default value is `None`.
        mo_occ_g : numpy.ndarray, optional
            Occupancies of the Green's function. If `None`, use
            `self.mo_occ_g`. Default value is `None`.

        Returns
        -------
        moments_occ : numpy.ndarray
            Moments of the occupied self-energy.
        moments_vir : numpy.ndarray
            Moments of the virtual self-energy.
        """

        # Get the orbitals
        if mo_energy_g is None:
            mo_energy_g = self.mo_energy_g
        if mo_occ_g is None:
            mo_occ_g = self.mo_occ_g

        # Setup dependent on diagonal SE
        q0, q1 = self.mpi_slice(mo_energy_g.size)
        if self.gw.diagonal_se:
            pq = "p"
            fproc = lambda x: np.diag(x)
        else:
            pq = "pq"
            fproc = lambda x: x

        # Initialise the moments
        nmo = eta.shape[-1]  # avoiding self.nmo for inheritence
        moments_occ = np.zeros((self.nmom_max + 1, nmo, nmo))
        moments_vir = np.zeros((self.nmom_max + 1, nmo, nmo))

        # Get the orders for the moments
        if eta_orders is None:
            eta_orders = np.arange(self.nmom_max + 1)
        eta_orders = np.asarray(eta_orders)

        for n in range(self.nmom_max + 1):
            if eta_orders.shape[0] == 1 and eta_orders[0] > n:
                pass
            else:
                # Get the binomial coefficients
                fp = scipy.special.binom(n, eta_orders)
                fh = fp * (-1) ** eta_orders

                # Construct the occupied moments for this order
                if np.any(mo_occ_g[q0:q1] > 0):
                    eo = np.power.outer(mo_energy_g[q0:q1][mo_occ_g[q0:q1] > 0], n - eta_orders)
                    to = util.einsum(f"t,kt,kt{pq}->{pq}", fh, eo, eta[mo_occ_g[q0:q1] > 0])
                    moments_occ[n] += fproc(to)

                # Construct the virtual moments for this order
                if np.any(mo_occ_g[q0:q1] == 0):
                    ev = np.power.outer(mo_energy_g[q0:q1][mo_occ_g[q0:q1] == 0], n - eta_orders)
                    tv = util.einsum(f"t,ct,ct{pq}->{pq}", fp, ev, eta[mo_occ_g[q0:q1] == 0])
                    moments_vir[n] += fproc(tv)

        # Sum over all processes
        moments_occ = mpi_helper.allreduce(moments_occ)
        moments_vir = mpi_helper.allreduce(moments_vir)

        # Numerical integration can lead to small non-hermiticity
        moments_occ = 0.5 * (moments_occ + moments_occ.swapaxes(1, 2).conj())
        moments_vir = 0.5 * (moments_vir + moments_vir.swapaxes(1, 2).conj())

        return moments_occ, moments_vir

    @logging.with_timer("Self-energy moments")
    @logging.with_status("Constructing self-energy moments")
    def build_se_moments(self, moments_dd=None, m0=None):
        """Build the moments of the self-energy via convolution.

        Parameters
        ----------
        moments_dd : numpy.ndarray, optional
            Moments of the density-density response.

        Returns
        -------
        moments_occ : numpy.ndarray
            Moments of the occupied self-energy.
        moments_vir : numpy.ndarray
            Moments of the virtual self-energy.
        """

        # Setup dependent on diagonal SE
        q0, q1 = self.mpi_slice(self.mo_energy_g.size)
        if self.gw.diagonal_se:
            eta = np.zeros((q1 - q0, self.nmo))
            pq = p = q = "p"
        else:
            eta = np.zeros((q1 - q0, self.nmo, self.nmo))
            pq, p, q = "pq", "p", "q"

        if self.d is None:
            self._build_d()

        if moments_dd is None:
            zeroth_mom = self.build_zeroth_moment(m0=m0)
            recursion_term = None

        # Initialise output moments
        moments_occ = np.zeros((self.nmom_max + 1, self.nmo, self.nmo))
        moments_vir = np.zeros((self.nmom_max + 1, self.nmo, self.nmo))

        # Get the moments in (aux|aux) and rotate to (mo|mo)
        for n in range(self.nmom_max + 1):
            if moments_dd is None:
                recursion_term, eta_aux = self.build_nth_dd_moment(n, recursion_term, zeroth_mom)
            else:
                eta_aux = np.dot(moments_dd[n], self.integrals.Lia.T)  # aux^2 o v
            eta_aux = mpi_helper.allreduce(eta_aux)
            for x in range(q1 - q0):
                Lp = self.integrals.Lpx[:, :, x]
                eta[x] = util.einsum(f"P{p},Q{q},PQ->{pq}", Lp, Lp, eta_aux) * 2.0

            # Construct the self-energy moments for this order only to
            # save memory
            moments_occ_n, moments_vir_n = self.convolve(eta[:, None], eta_orders=[n])
            moments_occ += moments_occ_n
            moments_vir += moments_vir_n

        return moments_occ, moments_vir

    @logging.with_timer("Dynamic polarizability moments")
    @logging.with_status("Constructing dynamic polarizability moments")
    def build_dp_moments(self):
        """Build the moments of the dynamic polarizability for optical spectra calculations.

        Returns
        -------
        moments : numpy.ndarray
            Moments of the dynamic polarizability.
        """

        p0, p1 = self.mpi_slice(self.nov)

        # Get the dipole matrices
        with self.gw.mol.with_common_orig((0, 0, 0)):
            dip = self.gw.mol.intor_symmetric("int1e_r", comp=3)

        # Rotate into ia basis
        ci = self.integrals.mo_coeff_w[:, self.integrals.mo_occ_w > 0]
        ca = self.integrals.mo_coeff_w[:, self.integrals.mo_occ_w == 0]
        dip = util.einsum("xpq,pi,qa->xia", dip, ci.conj(), ca)
        dip = dip.reshape(3, -1)

        # Get the density-density response moments
        moments_dd = self.build_dd_moments(m0=dip[:, p0:p1])

        # Get the moments of the dynamic polarizability
        moments_dp = util.einsum("px,nqx->npq", dip[:, p0:p1], moments_dd)

        return moments_dp

    @logging.with_timer("Inverse density-density moment")
    @logging.with_status("Constructing inverse density-density moment")
    def build_dd_moment_inv(self):
        r"""Build the first inverse (`n=-1`) moment of the density-density response.

        Returns
        -------
        moment : numpy.ndarray
            First inverse (`n=-1`) moment of the density-density
            response.

        Notes
        -----
        This is not the full `n=-1` moment, which is

        .. math::
            D^{-1} - D^{-1} V^\dagger (I + V D^{-1} V^\dagger)^{-1} \\
                    V D^{-1}

        but rather

        .. math:: (I + V D^{-1} V^\dagger)^{-1} V D^{-1}

        which ensures that the function scales properly. The final
        contractions are done when constructing the matrix-vector
        product.
        """

        # Initialise the moment
        p0, p1 = self.mpi_slice(self.nov)
        moment = np.zeros((self.nov, p1 - p0))

        # Construct energy differences
        d_full = lib.direct_sum(
            "a-i->ia",
            self.mo_energy_w[self.mo_occ_w == 0],
            self.mo_energy_w[self.mo_occ_w > 0],
        ).ravel()
        d = d_full[p0:p1]

        # Get the first inverse moment
        Liadinv = self.integrals.Lia / d[None]
        u = np.dot(Liadinv, self.integrals.Lia.T)
        u = mpi_helper.allreduce(u)
        u = np.linalg.inv(np.eye(self.naux) + u)
        moment = np.dot(u, Liadinv)

        return moment
