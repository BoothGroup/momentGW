import vayesta.rpa
import pyscf.lib
import numpy as np
from vayesta.core.util import dot

def get_tilde_dd_moms(mf, max_moment, Lpq=None, use_ri=True, npoints=48):
    """Given system specification, return the rpa dd moments contracted with cderi for the density fitted coulomb
    integrals."""
    rot = get_cderi_ph_rot(mf)

    if use_ri:
        moms = np.array(get_dd_moments_rirpa(mf, max_moment, rot, npoints=npoints, Lpq=Lpq))
    else:
        moms = np.array(get_dd_moments_rpa(mf, max_moment, rot))
    return moms

def get_cderi_ph_rot(mf):
    # Define the desired rotation of our particle-hole basis; we're assuming we aren't chunking the RI basis.
    # Get in AO basis.
    cderi = pyscf.lib.unpack_tril(next(mf.with_df.loop(blksize=mf.with_df.get_naoaux())))
    # Transform into particle-hole excitations.
    def get_ph(_cderi, co, cv):
        #einsum("npq,pi,qa->nia")
        temp = np.tensordot(np.tensordot(_cderi, co, [[1], [0]]), cv, [[1],[0]])
        return temp.reshape((temp.shape[0], -1))

    if len(mf.mo_coeff.shape) < 3:
        # RHF
        nocc = int(sum(mf.mo_occ) / 2)
        rot = get_ph(cderi, mf.mo_coeff[:, :nocc], mf.mo_coeff[:, nocc:])
        rot = np.concatenate([rot, rot], axis=1)
    else:
        # UHF
        nocc = int(sum(mf.mo_occ.T))
        rots = [get_ph(cderi, mf.mo_coeff[i, :, :nocc[i]], mf.mo_coeff[i, :, nocc[i]:]) for i in [0,1]]
        rot = np.concatenate(rots, axis=1)
    return rot


def get_dd_moments_rpa(mf, max_moment, rot):
    rpa = vayesta.rpa.ssRPA(mf)
    erpa = rpa.kernel()
    moms = rpa.gen_moms(max_moment)
    moms = [moms[x] for x in range(max_moment+1)]
    # Now need to rotate into desired basis.
    moms = [dot(rot, x, rot.T) for x in moms]
    return moms

def get_dd_moments_rirpa(mf, max_moment, rot, npoints, Lpq=None):
    myrirpa = vayesta.rpa.ssRIRPA(mf, Lpq=Lpq)

    moms = myrirpa.kernel_moms(max_moment, rot, npoints=npoints)[0]
    # Just need to project the RHS and we're done.
    return [dot(x, rot.T) for x in moms]
