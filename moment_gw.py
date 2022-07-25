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
from pyscf import __config__
import rpamoms

einsum = lib.einsum

def kernel(agw, nmom, mo_energy, mo_coeff, Lpq=None, orbs=None,
           vhf_df=False, verbose=logger.NOTE):
    '''Moment constrained GW

    Returns:
        Bool :      converged
        GF object:  gf (GW Greens function, in pole representation)
        GF object:  se (GW self-energy, in pole representation)
    '''
    mf = agw._scf
    if agw.frozen is None:
        frozen = 0
    else:
        frozen = agw.frozen
    assert frozen == 0

    if Lpq is None:
        Lpq = agw.ao2mo(mo_coeff)

    # FIXME: For the moment, 'orbs' doesn't do anything.
    if orbs is None:
        orbs = range(agw.nmo)
    else:
        raise NotImplementedError

    # v_xc
    v_mf = mf.get_veff() - mf.get_j()
    v_mf = reduce(numpy.dot, (mo_coeff.T, v_mf, mo_coeff))

    nocc = agw.nocc
    nmo = agw.nmo
    nvir = nmo - nocc
    naux = agw.with_df.get_naoaux()

    # v_hf from DFT/HF density
    if vhf_df: # and frozen == 0:
        # density fitting for vk
        vk = -einsum('Lni,Lim->nm',Lpq[:,:,:nocc],Lpq[:,:nocc,:])
    else:
        # exact vk without density fitting
        dm = mf.make_rdm1()
        rhf = scf.RHF(agw.mol)
        vk = rhf.get_veff(agw.mol,dm) - rhf.get_j(agw.mol,dm)
        vk = reduce(numpy.dot, (mo_coeff.T, vk, mo_coeff))

    # Get the moments of the screened Coulomb interaction, tild_eta.
    logger.debug(agw, "Computing the moments of the tild_eta (dd moments)")
    tild_etas = rpamoms.get_tilde_dd_moms(mf, nmom, use_ri=not agw.exact_dRPA)
    assert(tild_etas.shape==(nmom+1,naux,naux))

    if False:
        # As an independent sanity check for specific defs, 
        # we can compute them from pyscf (N^6)
        from pyscf import tdscf
        _tdscf = tdscf.dRPA(mf)
        _tdscf.nstates = nocc*nvir
        _tdscf.verbose = 0
        _tdscf.kernel()
        td_e = _tdscf.e
        td_xy = _tdscf.xy
        nexc = len(td_e)
        # factor of 2 for normalization, see tdscf/rhf.py
        td_xy = 2*np.asarray(td_xy) # (nexc, 2, nocc, nvir)
        # td_z is X+Y
        td_z = np.sum(td_xy, axis=1).reshape(nexc,nocc,nvir)
        td_en = np.zeros((nexc, nmom+1))
        for i in range(nmom+1):
            td_en[:,i] = np.power(td_e, i) # Raise RPA excitation energies to power
        # Compute moments of dd response
        etas = einsum('via,vn,vjb->iajbn',td_z, td_en, td_z)
        # Contract with Coulomb interaction to form the tilde eta values
        tild_etas_check = einsum('Qia,iajbn,Pjb->nQP', Lpq[:,:nocc,nocc:], etas, Lpq[:,:nocc,nocc:])
        assert(np.allclose(tild_etas, tild_etas_check))

    logger.debug(agw, "Contracting dd moments with second coulomb interaction")
    # TODO: If only orbital subset specified, constrain the range of q here
    X_ = einsum('Pqx,nQP->nqxQ',Lpq,tild_etas) # naux^2 nmo^2 nmom contraction
    # TODO: Check it isn't contracting over x index here, as this is contracted later?!
    tild_sigma = einsum('Qpx,nqxQ->npqx',Lpq,X_) # naux nmo^3 nmom contraction

    logger.debug(agw, "Forming particle and hole self-energy moments up to (and including) order {}".format(nmom))

    particle_se_moms = []
    hole_se_moms = []
    for n in range(nmom+1):
        mom_p = np.zeros((nmo,nmo))
        mom_h = np.zeros((nmo,nmo))
        for t in range(nmom+1):
            mom_h += scipy.special.binom(n,t) * (-1)**t * einsum('k,pqk->pq',np.power(mo_energy[:nocc],n-t), tild_sigma[t,:,:,:nocc])
            mom_p += scipy.special.binom(n,t) * einsum('c,pqc->pq',np.power(mo_energy[nocc:],n-t), tild_sigma[t,:,:,nocc:])
        particle_se_moms.append(mom_p)
        hole_se_moms.append(mom_h)

    se_static = vk - v_mf

    if agw.diag_sigma:
        # Approximate all moments by just their diagonal.
        # This should mean that the full frequency-dependent self-energy
        # is also diagonal.
        # Assuming that the quasiparticle solutions converge to the right
        # poles (i.e. se poles / aux energies are far from orbital energies)
        # then this should allow direct comparison to other GW
        # implementations that iteratively solve the diagonal qp equation.
        for i in range(len(particle_se_moms)):
            particle_se_moms[i] = np.diag(np.diag(particle_se_moms[i]))
            hole_se_moms[i] = np.diag(np.diag(hole_se_moms[i]))
        se_static = np.diag(np.diag(se_static))

    # We now have a list of hole and particle moments in hole_se_momes and particle_se_moms
    # We also have a 'static' part of the self energy, in se_static
    # Ollie...do your thing.
    # TO CHECK: Sign convention of the moments.
    gf_occ, se_occ = block_lanczos_se(se_static, hole_se_moms)
    gf_vir, se_vir = block_lanczos_se(se_static, particle_se_moms)
    se = combine(se_occ, se_vir)
    gf = combine(gf_occ, gf_vir)
    # FIXME need to get a chempot?
    conv = True

    for n, ref in enumerate(hole_se_moms):
        mom = se_occ.moment(n)
        logger.info(agw, "Error in hole moment %d: %.5g", n, np.max(np.abs(ref-mom)))

    for n, ref in enumerate(particle_se_moms):
        mom = se_vir.moment(n)
        logger.info(agw, "Error in particle moment %d: %.5g", n, np.max(np.abs(ref-mom)))

    gf_occ = gf.get_occupied()
    for n in range(min(5, gf_occ.naux)):
        en = gf_occ.energy[-(n+1)]
        vn = gf_occ.coupling[:, -(n+1)]
        qpwt = np.linalg.norm(vn)**2
        logger.note(agw, "IP energy level %d E = %.16g  QP weight = %0.6g", n, en, qpwt)

    gf_vir = gf.get_virtual()
    for n in range(min(5, gf_vir.naux)):
        en = gf_vir.energy[n]
        vn = gf_vir.coupling[:, n]
        qpwt = np.linalg.norm(vn)**2
        logger.note(agw, "EA energy level %d E = %.16g  QP weight = %0.6g", n, en, qpwt)

    # Then, return some object with the gf and se in an 'pole' / 'auxiliary' representation respectively

    return conv, gf, se


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
    gf : agf2.GreensFunction
        Green's function object
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

    h_aux = np.block([
        [se_static, v_aux],
        [v_aux.T, np.diag(e_aux)],
    ])

    e_gf, v_gf = np.linalg.eigh(h_aux)
    v_gf = v_gf[:solver.norb]

    se = SelfEnergy(e_aux, v_aux)
    gf = GreensFunction(e_gf, v_gf)

    return gf, se


class AGW(lib.StreamObject):

    # Whether to only consider the diagonal of the self-energy, for
    # comparison to other GW methods which iteratively solve the
    # diagonal quasiparticle equation
    diag_sigma = getattr(__config__, 'gw_gw_GW_diag_sigma', False)
    # Whether to exactly solve the frequency-integral for the dd moments
    # which is N^6, rather than NI at N^4
    exact_dRPA = getattr(__config__, 'gw_gw_GW_exact_dRPA', False)

    def __init__(self, mf, frozen=None):
        self.mol = mf.mol
        self._scf = mf
        self.verbose = self.mol.verbose
        self.stdout = self.mol.stdout
        self.max_memory = mf.max_memory

        self.frozen = frozen
        #TODO: implement frozen orbs
        if not (self.frozen is None or self.frozen == 0):
            raise NotImplementedError

        # moment-GW must use density fitting integrals for the N^4 algorithm
        if getattr(mf, 'with_df', None):
            self.with_df = mf.with_df
        else:
            self.with_df = df.DF(mf.mol)
            self.with_df.auxbasis = df.make_auxbasis(mf.mol, mp2fit=True)
        self._keys.update(['with_df'])

##################################################
# don't modify the following attributes, they are not input options
        self._nocc = None
        self._nmo = None
        # self.mo_energy: GW quasiparticle energy, not scf mo_energy
        self.mo_energy = None
        self.mo_coeff = mf.mo_coeff
        self.mo_occ = mf.mo_occ
        self.sigma = None
        self.gf = None

        keys = set(('diag_sigma', 'exact_dRPA'))
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
        logger.info(self, 'Use N^6 exact computation of self-energy moments = %s', self.exact_dRPA)
        logger.info(self, 'Use diagonal self-energy in QP eqn = %s', self.diag_sigma)
        return self

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

    def kernel(self, nmom=1, mo_energy=None, mo_coeff=None, Lpq=None, orbs=None, vhf_df=False, roots=10):
        """
        Input:
            nmom:      Particle and hole moment truncation of the self-energy
            mo_energy: MO energies of G0 (by default taken from mf)
            mo_coeff:  MO orbitals of G0 (by default taken from mf)
            Lpq:       3-index DF integrals (by default, constructed)
            orbs:      Restrict SE over these orbs (TODO)
            vhf_df:    Construct exchange matrix within DF approx? 
            roots:     Output the lowest-lying IPs/EAs with qp weights and orbital overlaps
        Output:
            gf object

        TODO: Pass in any other parameters for the NI?
        """
        if mo_coeff is None:
            mo_coeff = self._scf.mo_coeff
        if mo_energy is None:
            mo_energy = self._scf.mo_energy

        cput0 = (logger.process_clock(), logger.perf_counter())
        self.dump_flags()
        logger.info(self, 'Number of moments to compute in self-energy expansion = %s', nmom)
        self.converged, self.gf, self.sigma = \
                kernel(self, nmom, mo_energy, mo_coeff,
                       Lpq=Lpq, orbs=orbs, vhf_df=vhf_df, verbose=self.verbose)

        # TODO: Improve output to give the 'roots' lowest lying IP/EAs, qpwts and overlaps with MOs

        logger.timer(self, 'Moment GW', *cput0)
        return self.converged, self.gf, self.sigma

    def ao2mo(self, mo_coeff=None):
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
