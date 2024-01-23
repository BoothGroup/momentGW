"""
Construct TDA moments.
"""

import numpy as np
import scipy.special
from pyscf import lib

from momentGW import mpi_helper, util


class dTDA:
    """
    Compute the self-energy moments using dTDA.

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
        self.gw = gw
        self.nmom_max = nmom_max
        self.integrals = integrals

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

    def kernel(self, exact=False):
        """
        Run the polarizability calculation to compute moments of the
        self-energy.

        Parameters
        ----------
        exact : bool, optional
            If `True`, use the exact approach. Default value is `False`.

        Returns
        -------
        moments_occ : numpy.ndarray
            Moments of the occupied self-energy.
        moments_vir : numpy.ndarray
            Moments of the virtual self-energy.
        """

        lib.logger.info(
            self.gw,
            "Constructing %s moments (nmom_max = %d)",
            self.__class__.__name__,
            self.nmom_max,
        )

        if exact:
            moments_dd = self.build_dd_moments_exact()
        else:
            moments_dd = self.build_dd_moments()

        moments_occ, moments_vir = self.build_se_moments(moments_dd)

        return moments_occ, moments_vir

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

        cput0 = (lib.logger.process_clock(), lib.logger.perf_counter())
        lib.logger.info(self.gw, "Building density-density moments")
        lib.logger.debug(self.gw, "Memory usage: %.2f GB", self._memory_usage())

        naux = self.naux if m0 is None else m0.shape[0]
        p0, p1 = self.mpi_slice(self.nov)
        moments = np.zeros((self.nmom_max + 1, naux, p1 - p0))

        # Construct energy differences
        d = util.build_1h1p_energies(self.mo_energy_w, self.mo_occ_w).ravel()[p0:p1]

        # Get the zeroth order moment
        moments[0] = m0 if m0 is not None else self.integrals.Lia
        cput1 = lib.logger.timer(self.gw, "zeroth moment", *cput0)

        # Get the higher order moments
        for i in range(1, self.nmom_max + 1):
            moments[i] = moments[i - 1] * d[None]
            tmp = np.dot(moments[i - 1], self.integrals.Lia.T)
            tmp = mpi_helper.allreduce(tmp)
            moments[i] += np.dot(tmp, self.integrals.Lia) * 2.0
            del tmp
            cput1 = lib.logger.timer(self.gw, "moment %d" % i, *cput1)

        return moments

    build_dd_moments_exact = build_dd_moments

    def convolve(self, eta, eta_orders=None, mo_energy_g=None, mo_occ_g=None):
        """
        Handle the convolution of the moments of the Green's function
        and screened Coulomb interaction.

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

        nmo = eta.shape[-1]  # avoiding self.nmo for inheritence
        moments_occ = np.zeros((self.nmom_max + 1, nmo, nmo))
        moments_vir = np.zeros((self.nmom_max + 1, nmo, nmo))

        if eta_orders is None:
            eta_orders = np.arange(self.nmom_max + 1)
        eta_orders = np.asarray(eta_orders)

        for n in range(self.nmom_max + 1):
            fp = scipy.special.binom(n, eta_orders)
            fh = fp * (-1) ** eta_orders

            if np.any(mo_occ_g[q0:q1] > 0):
                eo = np.power.outer(mo_energy_g[q0:q1][mo_occ_g[q0:q1] > 0], n - eta_orders)
                to = lib.einsum(f"t,kt,kt{pq}->{pq}", fh, eo, eta[mo_occ_g[q0:q1] > 0])
                moments_occ[n] += fproc(to)

            if np.any(mo_occ_g[q0:q1] == 0):
                ev = np.power.outer(mo_energy_g[q0:q1][mo_occ_g[q0:q1] == 0], n - eta_orders)
                tv = lib.einsum(f"t,ct,ct{pq}->{pq}", fp, ev, eta[mo_occ_g[q0:q1] == 0])
                moments_vir[n] += fproc(tv)

        moments_occ = mpi_helper.allreduce(moments_occ)
        moments_vir = mpi_helper.allreduce(moments_vir)

        # Numerical integration can lead to small non-hermiticity
        moments_occ = 0.5 * (moments_occ + moments_occ.swapaxes(1, 2).conj())
        moments_vir = 0.5 * (moments_vir + moments_vir.swapaxes(1, 2).conj())

        return moments_occ, moments_vir

    def build_se_moments(self, moments_dd):
        """Build the moments of the self-energy via convolution.

        Parameters
        ----------
        moments_dd : numpy.ndarray
            Moments of the density-density response.

        Returns
        -------
        moments_occ : numpy.ndarray
            Moments of the occupied self-energy.
        moments_vir : numpy.ndarray
            Moments of the virtual self-energy.
        """

        cput0 = (lib.logger.process_clock(), lib.logger.perf_counter())
        lib.logger.info(self.gw, "Building self-energy moments")
        lib.logger.debug(self.gw, "Memory usage: %.2f GB", self._memory_usage())

        # Setup dependent on diagonal SE
        q0, q1 = self.mpi_slice(self.mo_energy_g.size)
        if self.gw.diagonal_se:
            eta = np.zeros((q1 - q0, self.nmo))
            pq = p = q = "p"
        else:
            eta = np.zeros((q1 - q0, self.nmo, self.nmo))
            pq, p, q = "pq", "p", "q"

        # Initialise output moments
        moments_occ = np.zeros((self.nmom_max + 1, self.nmo, self.nmo))
        moments_vir = np.zeros((self.nmom_max + 1, self.nmo, self.nmo))

        # Get the moments in (aux|aux) and rotate to (mo|mo)
        for n in range(self.nmom_max + 1):
            eta_aux = np.dot(moments_dd[n], self.integrals.Lia.T)  # aux^2 o v
            eta_aux = mpi_helper.allreduce(eta_aux)
            for x in range(q1 - q0):
                Lp = self.integrals.Lpx[:, :, x]
                eta[x] = lib.einsum(f"P{p},Q{q},PQ->{pq}", Lp, Lp, eta_aux) * 2.0

            # Construct the self-energy moments for this order only
            # to save memory
            moments_occ_n, moments_vir_n = self.convolve(eta[:, None], eta_orders=[n])
            moments_occ += moments_occ_n
            moments_vir += moments_vir_n

        lib.logger.timer(self.gw, "constructing SE moments", *cput0)

        return moments_occ, moments_vir

    def build_dp_moments(self):
        """
        Build the moments of the dynamic polarizability for optical
        spectra calculations.

        Returns
        -------
        moments : numpy.ndarray
            Moments of the dynamic polarizability.
        """

        cput0 = (lib.logger.process_clock(), lib.logger.perf_counter())
        lib.logger.info(self.gw, "Building dynamic polarizability moments")
        lib.logger.debug(self.gw, "Memory usage: %.2f GB", self._memory_usage())

        p0, p1 = self.mpi_slice(self.nov)

        # Get the dipole matrices
        with self.gw.mol.with_common_orig((0, 0, 0)):
            dip = self.gw.mol.intor_symmetric("int1e_r", comp=3)

        # Rotate into ia basis
        ci = self.integrals.mo_coeff_w[:, self.integrals.mo_occ_w > 0]
        ca = self.integrals.mo_coeff_w[:, self.integrals.mo_occ_w == 0]
        dip = lib.einsum("xpq,pi,qa->xia", dip, ci.conj(), ca)
        dip = dip.reshape(3, -1)

        # Get the density-density response moments
        moments_dd = self.build_dd_moments(m0=dip[:, p0:p1])

        # Get the moments of the dynamic polarizability
        moments_dp = lib.einsum("px,nqx->npq", dip[:, p0:p1], moments_dd)

        lib.logger.timer(self.gw, "moments", *cput0)

        return moments_dp

    def build_dd_moment_inv(self):
        r"""
        Build the first inverse (`n=-1`) moment of the density-density
        response.

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

        cput0 = (lib.logger.process_clock(), lib.logger.perf_counter())
        lib.logger.info(self.gw, "Building first inverse density-density moment")
        lib.logger.debug(self.gw, "Memory usage: %.2f GB", self._memory_usage())

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

        lib.logger.timer(self.gw, "inverse moment", *cput0)

        return moment

    def _memory_usage(self):
        """Return the current memory usage in GB."""
        return lib.current_memory()[0] / 1e3

    @property
    def nmo(self):
        """Number of MOs."""
        return self.gw.nmo

    @property
    def naux(self):
        """Number of auxiliaries."""
        return self.integrals.naux

    @property
    def nov(self):
        """Number of ov states in the screened Coulomb interaction."""
        return np.sum(self.mo_occ_w > 0) * np.sum(self.mo_occ_w == 0)

    def mpi_slice(self, n):
        """
        Return the start and end index for the current process for total
        size `n`.

        Parameters
        ----------
        n : int
            Total size.

        Returns
        -------
        p0 : int
            Start index for current process.
        p1 : int
            End index for current process.
        """
        return list(mpi_helper.prange(0, n, n))[0]

    def mpi_size(self, n):
        """
        Return the number of states in the current process for total size
        `n`.

        Parameters
        ----------
        n : int
            Total size.

        Returns
        -------
        size : int
            Number of states in current process.
        """
        p0, p1 = self.mpi_slice(n)
        return p1 - p0
