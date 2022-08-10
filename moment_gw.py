'''
Spin-restricted G0W0 via self-energy moment constraints

This implementation should be N^4, if the screened Coulomb moments
are computed via numerical integration, or N^6 otherwise. It should
converge to exact full-frequency integration as the number of self-energy
moments computed increases.

Note that the 'quasi-particle' equation is solved exactly via full
Dyson inversion in N^3 time, without diagonal self-energy approximation,
and will return all possible Greens function poles (which may be more
than the original mean-field), along with their quasi-particle weights.
This avoids the traditional diagonal self-energy approximation and assumption
that self-energy poles are far from MO energies.
'''

from functools import reduce
import scipy.special
import numpy
import numpy as np
import h5py

from pyscf import lib
from pyscf.lib import logger
from pyscf.ao2mo import _ao2mo
from pyscf import df, scf
from pyscf.mp.mp2 import get_nocc, get_nmo, get_frozen_mask
from pyscf.agf2.aux_space import GreensFunction, SelfEnergy, combine
from pyscf.agf2 import chempot
from pyscf import __config__
import rpamoms

einsum = lib.einsum

DEBUG = True

# TODO:
# Make nmom notation consistent with AGF2
# orbs argument
# frozen argument


def kernel(agw, nmom, mo_energy, mo_coeff, Lpq=None, orbs=None,
           vhf_df=False, npoints=48, verbose=logger.NOTE):
    """Moment-constrained GW. Returns the Green's function and self-
    energy as objects using the `GreensFunction` and `SelfEnergy`
    objects from the `agf2` module.

    Parameters
    ----------
    agw : AGW
        AGW object.
    nmom : int
        Maximum moment number to calculate.
    mo_energy : numpy.ndarray
        Molecular orbital energies.
    mo_coeff : numpy.ndarray
        Molecular orbital coefficients.
    Lpq : np.ndarray, optional
        Density-fitted ERI tensor. If None, generate from `agw.ao2mo`.
        Default value is None.
    orbs : list of int, optional
        List of orbitals to include in GW calculation. If None,
        include all orbitals. Default value is None.
    vhf_df : bool, optional
        If True, calculate the static self-energy directly from `Lpq`.
        Default value is False.

    Returns
    -------
    conv : bool
        Convergence flag. Always True for AGW, returned for
        compatibility with other GW methods.
    gf : pyscf.agf2.GreensFunction
        Green's function object
    se : pyscf.agf2.SelfEnergy
        Self-energy object
    """

    mf = agw._scf
    if agw.frozen is None:
        frozen = 0
    else:
        frozen = agw.frozen
    assert frozen == 0

    if Lpq is None:
        Lpq = agw.ao2mo(mo_coeff)

    if orbs is None:
        orbs = range(agw.nmo)
    else:
        raise NotImplementedError

    nocc = agw.nocc
    nmo = agw.nmo
    nvir = nmo - nocc
    naux = agw.with_df.get_naoaux()

    se_static = agw.build_se_static(Lpq=Lpq, mo_coeff=mo_coeff, vhf_df=vhf_df)
    hole_se_moms, particle_se_moms = \
            agw.build_se_moments(nmom, Lpq=Lpq, mo_energy=mo_energy, mo_coeff=mo_coeff, npoints=npoints)

    if agw.diag_sigma:
        # TODO move this to docstring:
        # Approximate all moments by just their diagonal.
        # This should mean that the full frequency-dependent self-energy
        # is also diagonal.
        # Assuming that the quasiparticle solutions converge to the right
        # poles (i.e. se poles / aux energies are far from orbital energies)
        # then this should allow direct comparison to other GW
        # implementations that iteratively solve the diagonal qp equation.
        pass
    
    gf, se = agw.solve_dyson(hole_se_moms, particle_se_moms, se_static, mo_energy=mo_energy)
    conv = True

    se_occ = se.get_occupied()
    for n, ref in enumerate(hole_se_moms):
        mom = se_occ.moment(n)
        err = np.max(np.abs(ref-mom)) / np.max(np.abs(ref))
        logger.debug(agw, "Error in hole moment %d: %.5g", n, err)

    se_vir = se.get_virtual()
    for n, ref in enumerate(particle_se_moms):
        mom = se_vir.moment(n)
        err = np.max(np.abs(ref-mom)) / np.max(np.abs(ref))
        logger.debug(agw, "Error in particle moment %d: %.5g", n, err)

    return conv, gf, se


def build_se_static(agw, Lpq=None, vhf_df=False, mo_coeff=None):
    """Build the static part of the self-energy.

    Parameters
    ----------
    agw : AGW
        AGW object.
    Lpq : np.ndarray, optional
        Density-fitted ERI tensor. If None, generate from `agw.ao2mo`.
        Default value is None.
    vhf_df : bool, optional
        If True, calculate the static self-energy directly from `Lpq`.
        Default value is False.
    mo_coeff : numpy.ndarray, optional
        Molecular orbital coefficients. If None, use array from
        `agw._scf`. Default value is None.

    Returns
    -------
    se_static : np.ndarray
        Static part of the self-energy. If `agw.diag_sigma`,
        non-diagonal elements are set to zero.
    """

    if mo_coeff is None:
        mo_coeff = agw._scf.mo_coeff
    if Lpq is None and vhf_df:
        Lpq = agw.ao2mo(mo_coeff)

    v_mf = agw._scf.get_veff() - agw._scf.get_j()
    v_mf = einsum("pq,pi,qj->ij", v_mf, mo_coeff, mo_coeff)

    # v_hf from DFT/HF density
    if vhf_df:
        sc = np.dot(agw._scf.get_ovlp(), mo_coeff)
        dm = einsum("pq,pi,qj->ij", agw._scf.make_rdm1(mo_coeff=mo_coeff), sc, sc)
        tmp = einsum("Qik,kl->Qil", Lpq, dm)
        vk = -einsum("Qil,Qlj->ij", tmp, Lpq) * 0.5
    else:
        dm = agw._scf.make_rdm1(mo_coeff=mo_coeff)
        vk = scf.hf.SCF.get_veff(agw._scf, agw.mol, dm) - scf.hf.SCF.get_j(agw._scf, agw.mol, dm)
        vk = einsum("pq,pi,qj->ij", vk, mo_coeff, mo_coeff)

    se_static = vk - v_mf

    if agw.diag_sigma:
        se_static = np.diag(np.diag(se_static))

    return se_static


def build_se_moments(agw, nmom, Lpq=None, mo_energy=None, mo_coeff=None, npoints=48, debug=DEBUG):
    """Build the density-density response moments.

    Parameters
    ----------
    agw : AGW
        AGW object.
    nmom : int
        Maximum moment number to calculate.
    Lpq : np.ndarray, optional
        Density-fitted ERI tensor. If None, generate from `agw.ao2mo`.
        Default value is None.
    mo_energy : numpy.ndarray, optional
        Molecular orbital energies. If None, use array from `agw._scf`.
        Default value is None.
    mo_coeff : numpy.ndarray, optional
        Molecular orbital coefficients. If None, use array from
        `agw._scf`. Default value is None.

    Returns
    -------
    hole_moms : np.ndarray
        Moments of the hole self-energy. If `agw.diag_sigma`,
        non-diagonal elements are set to zero.
    part_moms : np.ndarray
        Moments of the particle self-energy. If `agw.diag_sigma`,
        non-diagonal elements are set to zero.
    """

    if mo_energy is None:
        mo_energy = agw._scf.mo_energy
    if Lpq is None:
        Lpq = agw.ao2mo(mo_coeff)

    nmo = agw.nmo
    nocc = agw.nocc

    logger.debug(agw, "Building moments up to nmom = %d", nmom)
    logger.debug(agw, "Computing the moments of the tild_eta (~ screened coulomb moments)")
    tild_etas = rpamoms.get_tilde_dd_moms(agw._scf, nmom, Lpq=Lpq, use_ri=not agw.exact_dRPA, npoints=npoints)

    logger.debug(agw, "Contracting dd moments with second coulomb interaction")
    if agw.diag_sigma:
        # Simplifications due to only constructing the diagonal self-energy
        tild_sigma = np.zeros((nmo, nmom+1, nmo))
        for x in range(nmo):
            Lpx = Lpq[:, :, x]
            tild_sigma[x] = einsum("Pp,Qp,nQP->np", Lpx, Lpx, tild_etas)

        logger.debug(agw, "Forming particle and hole self-energy")
        part_moms = []
        hole_moms = []
        moms = np.arange(nmom+1)
        for n in range(nmom+1):
            fp = scipy.special.binom(n, moms)
            fh = fp * (-1)**moms
            e = np.power.outer(mo_energy, n-moms)
            th = einsum("t,kt,ktp->p", fh, e[:nocc], tild_sigma[:nocc])
            tp = einsum("t,ct,ctp->p", fp, e[nocc:], tild_sigma[nocc:])
            hole_moms.append(np.diag(th))
            part_moms.append(np.diag(tp))

    else:
        tild_sigma = np.zeros((nmo, nmom+1, nmo, nmo))
        for x in range(nmo):
            Lpx = Lpq[:, :, x]
            tild_sigma[x] = einsum("Pq,Qp,nQP->npq", Lpx, Lpx, tild_etas)

        logger.debug(agw, "Forming particle and hole self-energy")
        part_moms = []
        hole_moms = []
        moms = np.arange(nmom+1)
        for n in range(nmom+1):
            fp = scipy.special.binom(n, moms)
            fh = fp * (-1)**moms
            e = np.power.outer(mo_energy, n-moms)
            th = einsum("t,kt,ktpq->pq", fh, e[:nocc], tild_sigma[:nocc])
            tp = einsum("t,ct,ctpq->pq", fp, e[nocc:], tild_sigma[nocc:])
            hole_moms.append(th)
            part_moms.append(tp)

    #if agw.diag_sigma:
    #    hole_moms = [np.diag(np.diag(t)) for t in hole_moms]
    #    part_moms = [np.diag(np.diag(t)) for t in part_moms]

    if debug:
        # Log the definiteness of the moments
        mmin = lambda x: ("%12.6g" % min(x)) if len(x) else ""
        mmax = lambda x: ("%12.6g" % max(x)) if len(x) else ""
        logger.debug(agw, "%12s %12s %12s %12s %12s", "Moment", "min neg", "max neg", "min pos", "max pos")
        for n in range(nmom+1):
            w = np.linalg.eigvalsh(hole_moms[n])
            vals = (mmin(w[w<0]), mmax(w[w<0]), mmin(w[w>=0]), mmax(w[w>=0]))
            logger.debug(agw, "hole %-7d %12s %12s %12s %12s", n, *vals)
        for n in range(nmom+1):
            w = np.linalg.eigvalsh(part_moms[n])
            vals = (mmin(w[w<0]), mmax(w[w<0]), mmin(w[w>=0]), mmax(w[w>=0]))
            logger.debug(agw, "part %-7d %12s %12s %12s %12s", n, *vals)

    return hole_moms, part_moms


def solve_dyson(agw, hole_moms, part_moms, se_static, mo_energy=None):
    """Solve the Dyson equation due to a self-energy resulting from
    a list of hole and particle moments, along with a static
    contribution to the self-energy.

    Also finds a chemical potential which best staisfies the physical
    number of electrons. If `agw.optimise_chempot`, this will shift
    the self-energy poles relative to the Green's function, which may
    be considered a partial self-consistency. Otherwise, just find a
    chemical potential according to Aufbau principal and do not allow
    relative difference in chemical potentials in SE and GF.

    Parameters
    ----------
    agw : AGW
        AGW object.
    hole_moms : np.ndarray
        Moments of the hole self-energy.
    part_moms : np.ndarray
        Moments of the particle self-energy.
    mo_energy : numpy.ndarray, optional
        Molecular orbital energies. If None, use array from `agw._scf`.
        Default value is None.

    Returns
    -------
    gf : agf2.GreensFunction
        Green's function object
    se : agf2.SelfEnergy
        Self-energy object
    """

    if mo_energy is None:
        mo_energy = agw._scf.mo_energy

    fock = np.diag(mo_energy) + se_static

    se_occ = block_lanczos_se(fock, hole_moms)
    se_vir = block_lanczos_se(fock, part_moms)
    se = combine(se_occ, se_vir)

    if agw.optimise_chempot:
        # Shift the self-energy poles w.r.t the Green's function
        se, opt = chempot.minimize_chempot(se, fock, agw.mol.nelectron)

    gf = se.get_greens_function(fock)

    cpt, error = chempot.binsearch_chempot((gf.energy, gf.coupling), gf.nphys, agw.mol.nelectron)
    se.chempot = cpt
    gf.chempot = cpt
    logger.info(agw, "Error in number of electrons: %.5g", error)

    return gf, se


def block_lanczos_se(se_static, se_moms):
    """Transform a set of moments of the self-energy to a pole
    representation.

    Parameters
    ----------
    se_static : np.ndarray (n, n)
        Static part of the self-energy
    se_moms : np.ndarray (k, n, n)
        Moments of the self-energy, for j iterations of block Lanczos
        the first k=2*j+2 moments are required.

    Returns
    -------
    se : agf2.SelfEnergy
        Self-energy object
    """

    try:
        from dyson import BlockLanczosSymmSE
    except:
        # TODO implement this here so there's no weird dependency
        raise ValueError(
                "Missing dependency: "
                "https://github.com/obackhouse/dyson-compression"
        )

    solver = BlockLanczosSymmSE(se_static, se_moms)
    e_aux, v_aux = solver.get_auxiliaries()

    se = SelfEnergy(e_aux, v_aux)

    return se


class AGW(lib.StreamObject):
    """Moment-conserved GW.

    Parameters
    ----------
    mf : pyscf.scf.SCF
        PySCF mean-field object.
    frozen : int or tuple of int, optional
        Frozen orbitals. Default value is None.

    Attributes
    ----------
    diag_sigma : bool
        If True, assume a diagonal approximation in the self-energy.
        Default value is False.
    exact_dRPA : bool
        If True, exactly solve the frequency integral for the DD
        moments. Exact solution scales as N^6 in system sizes, while
        the numerical integration scales as N^4. Default value is
        False.
    optimise_chempot : bool
        If True, optimise the number of electrons by applying a
        constant shift in the poles of the self-energy, thereby
        applying a different relative chemical potential in the self-
        energy and Green's functions. Default value is False.
    gf : pyscf.agf2.GreensFunction
        Green's function object
    sigma : pyscf.agf2.SelfEnergy
        Self-energy object
    """

    diag_sigma = getattr(__config__, "gw_gw_GW_diag_sigma", False)
    exact_dRPA = getattr(__config__, "gw_gw_GW_exact_dRPA", False)
    optimise_chempot = getattr(__config__, "gw_gw_GW_optimise_chempot", False)

    def __init__(self, mf, frozen=None):
        self.mol = mf.mol
        self._scf = mf
        self.verbose = self.mol.verbose
        self.stdout = self.mol.stdout
        self.max_memory = mf.max_memory

        self.frozen = frozen
        if not (self.frozen is None or self.frozen == 0):
            raise NotImplementedError

        # moment-GW must use density fitting integrals for the N^4 algorithm
        if getattr(mf, 'with_df', None):
            self.with_df = mf.with_df
        else:
            self.with_df = df.DF(mf.mol)
            self.with_df.auxbasis = df.make_auxbasis(mf.mol, mp2fit=True)

##################################################
# don't modify the following attributes, they are not input options
        self.mo_coeff = mf.mo_coeff
        self.mo_occ = mf.mo_occ
        self._nocc = None
        self._nmo = None
        self.sigma = None
        self.gf = None

        keys = set(("diag_sigma", "exact_dRPA", "optimise_chempot"))
        self._keys = set(self.__dict__.keys()).union(keys)

    def dump_flags(self):
        log = logger.Logger(self.stdout, self.verbose)
        log.info('')
        log.info('******** %s ********', self.__class__)
        log.info('method = %s', self.__class__.__name__)
        nocc = self.nocc
        nvir = self.nmo - nocc
        log.info('Moment-constrained GW nocc = %d, nvir = %d', nocc, nvir)
        if self.frozen is not None:
            log.info('frozen = %s', self.frozen)
        log.info('Use N^6 exact computation of self-energy moments = %s', self.exact_dRPA)
        log.info('Use diagonal self-energy in QP eqn = %s', self.diag_sigma)
        return self

    @property
    def mo_energy(self):
        # TODO should we prune low-weighted energies?
        return self.gf.mo_energy

    @property
    def nocc(self):
        return self.get_nocc()
    @nocc.setter
    def nocc(self, n):
        self._nocc = n

    @property
    def nmo(self):
        return self.get_nmo()
    @nmo.setter
    def nmo(self, n):
        self._nmo = n

    get_nocc = get_nocc
    get_nmo = get_nmo
    get_frozen_mask = get_frozen_mask

    build_se_static = build_se_static
    build_se_moments = build_se_moments
    solve_dyson = solve_dyson

    def kernel(self, nmom=1, mo_energy=None, mo_coeff=None, Lpq=None, orbs=None, vhf_df=False, roots=10, npoints=48):
        __doc__ = kernel.__doc__

        if mo_coeff is None:
            mo_coeff = self._scf.mo_coeff
        if mo_energy is None:
            mo_energy = self._scf.mo_energy

        cput0 = (logger.process_clock(), logger.perf_counter())
        self.dump_flags()
        logger.info(self, 'Number of moments to compute in self-energy expansion = %s', nmom)

        self.converged, self.gf, self.sigma = \
                kernel(self, nmom, mo_energy, mo_coeff,
                       Lpq=Lpq, orbs=orbs, vhf_df=vhf_df, npoints=npoints, verbose=self.verbose)

        gf_occ = self.gf.get_occupied()
        for n in range(min(roots, gf_occ.naux)):
            en = gf_occ.energy[-(n+1)]
            vn = gf_occ.coupling[:, -(n+1)]
            qpwt = np.linalg.norm(vn)**2
            logger.note(self, "IP energy level %d E = %.16g  QP weight = %0.6g", n, en, qpwt)

        gf_vir = self.gf.get_virtual()
        for n in range(min(roots, gf_vir.naux)):
            en = gf_vir.energy[n]
            vn = gf_vir.coupling[:, n]
            qpwt = np.linalg.norm(vn)**2
            logger.note(self, "EA energy level %d E = %.16g  QP weight = %0.6g", n, en, qpwt)

        logger.timer(self, 'Moment GW', *cput0)

        return self.converged, self.gf, self.sigma

    def ao2mo(self, mo_coeff=None):
        """Get MO basis density-fitted integrals.
        """

        if mo_coeff is None:
            mo_coeff = self.mo_coeff
        nmo = self.nmo
        naux = self.with_df.get_naoaux()
        mem_incore = (2*nmo**2*naux) * 8/1e6
        mem_now = lib.current_memory()[0]

        mo = numpy.asarray(mo_coeff, order='F')
        ijslice = (0, nmo, 0, nmo)
        Lpq = None
        if (mem_incore + mem_now < 0.99*self.max_memory) or self.mol.incore_anyway:
            Lpq = _ao2mo.nr_e2(self.with_df._cderi, mo, ijslice, aosym='s2', out=Lpq)
            return Lpq.reshape(naux,nmo,nmo)
        else:
            logger.warn(self, 'Memory may not be enough!')
            raise NotImplementedError

del DEBUG


if __name__ == '__main__':
    from pyscf import gto, dft
    mol = gto.Mole()
    mol.verbose = 4
    mol.atom = [
        [8 , (0. , 0.     , 0.)],
        [1 , (0. , -0.7571 , 0.5861)],
        [1 , (0. , 0.7571 , 0.5861)]]
    mol.basis = 'def2-svp'
    mol.build()

    mf = dft.RKS(mol)
    mf = mf.density_fit()
    mf.xc = 'pbe'
    mf.kernel()

    nocc = mol.nelectron//2
    nmo = mf.mo_energy.size
    nvir = nmo-nocc

    gw = AGW(mf)
    gw.exact_dRPA = True
    gw.kernel(nmom=5)
