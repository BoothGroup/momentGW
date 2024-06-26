"""
Spin-unrestricted quasiparticle self-consistent GW via self-energy
moment constraints for periodic systems.
"""

import numpy as np

from momentGW import util
from momentGW.pbc.qsgw import qsKGW
from momentGW.pbc.uhf.evgw import evKUGW
from momentGW.pbc.uhf.gw import KUGW
from momentGW.qsgw import qsGW
from momentGW.uhf.qsgw import qsUGW


class qsKUGW(KUGW, qsKGW, qsUGW):
    """
    Spin-unrestricted quasiparticle self-consistent GW via self-energy
    moment constraints for periodic systems.

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
    fc : bool, optional
        If `True`, apply finite size corrections. Default value is
        `False`.
    max_cycle : int, optional
        Maximum number of iterations. Default value is `50`.
    max_cycle_qp : int, optional
        Maximum number of iterations in the quasiparticle equation
        loop. Default value is `50`.
    conv_tol : float, optional
        Convergence threshold in the change in the HOMO and LUMO.
        Default value is `1e-8`.
    conv_tol_moms : float, optional
        Convergence threshold in the change in the moments. Default
        value is `1e-8`.
    conv_tol_qp : float, optional
        Convergence threshold in the change in the density matrix in
        the quasiparticle equation loop. Default value is `1e-8`.
    conv_logical : callable, optional
        Function that takes an iterable of booleans as input indicating
        whether the individual `conv_tol`, `conv_tol_moms`,
        `conv_tol_qp` have been satisfied, respectively, and returns a
        boolean indicating overall convergence. For example, the
        function `all` requires both metrics to be met, and `any`
        requires just one. Default value is `all`.
    diis_space : int, optional
        Size of the DIIS extrapolation space. Default value is `8`.
    diis_space_qp : int, optional
        Size of the DIIS extrapolation space in the quasiparticle
        loop. Default value is `8`.
    damping : float, optional
        Damping parameter. Default value is `0.0`.
    eta : float, optional
        Small value to regularise the self-energy. Default value is
        `1e-1`.
    srg : float, optional
        If non-zero, use the similarity renormalisation group approach
        of Marie and Loos in place of the `eta` regularisation. For
        value recommendations refer to their paper (arXiv:2303.05984).
        Default value is `0.0`.
    solver : BaseGW, optional
        Solver to use to obtain the self-energy. Compatible with any
        `BaseGW`-like class. Default value is `momentGW.gw.GW`.
    solver_options : dict, optional
        Keyword arguments to pass to the solver. Default value is an
        empty `dict`.
    """

    _defaults = util.dict_union(KUGW._defaults, qsKGW._defaults, qsUGW._defaults)
    _defaults["solver"] = KUGW

    check_convergence = evKUGW.check_convergence

    @property
    def name(self):
        """Get the method name."""
        polarizability = self.polarizability.upper().replace("DTDA", "dTDA").replace("DRPA", "dRPA")
        return f"{polarizability}-qsKUGW"

    @staticmethod
    def project_basis(matrix, ovlp, mo1, mo2):
        """
        Project a matrix from one basis to another.

        Parameters
        ----------
        matrix : numpy.ndarray or tuple of dyson.Lehmann
            Matrix to project at each k-point for each spin channel. Can
            also be a tuple of `dyson.Lehmann` objects, in which case the
            `couplings` attributes are projected.
        ovlp : numpy.ndarray
            Overlap matrix in the shared (AO) basis at each k-point.
        mo1 : numpy.ndarray
            First basis, rotates from the shared (AO) basis into the
            basis of `matrix` at each k-point for each spin channel.
        mo2 : numpy.ndarray
            Second basis, rotates from the shared (AO) basis into the
            desired basis of the output at each k-point for each spin
            channel.

        Returns
        -------
        proj : numpy.ndarray or tuple of dyson.Lehmann
            Matrix projected into the desired basis at each k-point
            for each spin channel.
        """

        # Build the projection matrix
        proj = util.einsum("k...pq,sk...pi,sk...qj->sk...ij", ovlp, np.conj(mo1), mo2)

        # Project the matrix
        if isinstance(matrix, np.ndarray):
            projected_matrix = util.einsum(
                "sk...pq,sk...pi,sk...qj->sk...ij", matrix, np.conj(proj), proj
            )
        else:
            projected_matrix = [[], []]
            for s, ms in enumerate(matrix):
                for k, m in enumerate(ms):
                    coupling = util.einsum("pk,pi->ik", m.couplings, np.conj(proj[s][k]))
                    projected_m = m.copy()
                    projected_m.couplings = coupling
                    projected_matrix[s].append(projected_m)

        return projected_matrix

    @staticmethod
    def self_energy_to_moments(se, nmom_max):
        """
        Return the hole and particle moments for a self-energy.

        Parameters
        ----------
        se : tuple of dyson.Lehmann
            Self-energy to compute the moments of at each k-point
            for each spin channel.

        Returns
        -------
        th : numpy.ndarray
            Hole moments at each k-point for each spin channel.
        tp : numpy.ndarray
            Particle moments at each k-point for each spin channel.
        """
        th = np.array([[s.occupied().moment(range(nmom_max + 1)) for s in ses] for ses in se])
        tp = np.array([[s.virtual().moment(range(nmom_max + 1)) for s in ses] for ses in se])
        return th, tp

    def build_static_potential(self, mo_energy, se):
        """
        Build the static potential approximation to the self-energy.

        Parameters
        ----------
        mo_energy : numpy.ndarray
            Molecular orbital energies at each k-point for each spin
            channel.
        se : tuple of dyson.Lehmann
            Self-energy to approximate at each k-point for each spin
            channel.

        Returns
        -------
        se_qp : numpy.ndarray
            Static potential approximation to the self-energy at each
            k-point for each spin channel.
        """
        return np.array(
            [
                [qsGW.build_static_potential(self, mo, s) for mo, s in zip(mos, ses)]
                for mos, ses in zip(mo_energy, se)
            ]
        )
