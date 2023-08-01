"""
Construct RPA moments with periodic boundary conditions.
"""

import numpy as np
from pyscf import lib
from pyscf.pbc.lib.kpts_helper import get_kconserv

from momentGW.pbc import df


def build_se_moments_drpa_exact(
    gw,
    nmom_max,
    Lpq,
    Lia,
    mo_energy=None,
):
    """
    Compute the self-energy moments using exact dRPA.  Scales as
    the sixth power of the number of orbitals.

    Parameters
    ----------
    gw : BaseKGW
        GW object.
    nmom_max : int
        Maximum moment number to calculate.
    Lpq : numpy.ndarray
        Density-fitted ERI tensor. `p` is in the basis of MOs, `q` is
        in the basis of the Green's function.
    Lia : numpy.ndarray
        Density-fitted ERI tensor for the occupied-virtual slice. `i`
        and `a` are in the basis of the screened Coulomb interaction.
    mo_energy : numpy.ndarray, optional
        Molecular orbital energies.  Default value is that of
        `gw._scf.mo_energy`.

    Returns
    -------
    se_moments_hole : numpy.ndarray
        Moments of the hole self-energy at each k-point. If
        `self.diagonal_se`, non-diagonal elements are set to zero.
    se_moments_part : numpy.ndarray
        Moments of the particle self-energy at each k-point. If
        `self.diagonal_se`, non-diagonal elements are set to zero.
    """

    if mo_energy is None:
        mo_energy = gw._scf.mo_energy

    nkpts = gw.nkpts
    nmo = gw.nmo
    nocc = np.array(gw.nocc)
    nov = nocc * (nmo - nocc)

    hole_moms = np.zeros((nkpts, nmom_max + 1, nmo, nmo), dtype=np.complex128)
    part_moms = np.zeros((nkpts, nmom_max + 1, nmo, nmo), dtype=np.complex128)

    scaled_kpts = gw.mol.get_scaled_kpts(gw.kpts)
    scaled_kpts -= scaled_kpts[0]
    kpt_dict = {df.hash_array(kpt): k for k, kpt in enumerate(scaled_kpts)}

    def wrap_around(kpt):
        # Handle wrap around for a single scaled k-point
        kpt[kpt >= (1.0 - df.KPT_DIFF_TOL)] -= 1.0
        return kpt

    # For now
    assert len(set(nocc)) == 1

    blocks = {}
    for q in range(nkpts):
        for ki in range(nkpts):
            for kj in range(nkpts):
                transfer = scaled_kpts[q] + scaled_kpts[ki] - scaled_kpts[kj]
                conserved = np.linalg.norm(np.round(transfer) - transfer) < 1e-12
                if conserved:
                    ki_p_q = kpt_dict[df.hash_array(wrap_around(scaled_kpts[ki] + scaled_kpts[q]))]
                    kj_p_q = kpt_dict[df.hash_array(wrap_around(scaled_kpts[kj] + scaled_kpts[q]))]

                    ei = mo_energy[ki][: nocc[ki]]
                    ea = mo_energy[ki_p_q][nocc[ki_p_q] :]

                    Via = Lpq[ki, ki_p_q, :, : nocc[ki], nocc[ki_p_q] :]
                    Vjb = Lpq[kj, kj_p_q, :, : nocc[kj], nocc[kj_p_q] :]
                    Vbj = Lpq[kj_p_q, kj, :, nocc[kj_p_q] :, : nocc[kj]]
                    iajb = lib.einsum("Lia,Ljb->iajb", Via, Vjb)
                    iabj = lib.einsum("Lia,Lbj->iajb", Via, Vbj)

                    blocks["a", q, ki, kj] = np.diag(
                        (ea[:, None] - ei[None]).ravel().astype(iajb.dtype)
                    )
                    blocks["a", q, ki, kj] += iabj.reshape(blocks["a", q, ki, kj].shape)
                    blocks["b", q, ki, kj] = iajb.reshape(blocks["a", q, ki, kj].shape)

    z = np.zeros((nov[0], nov[0]), dtype=np.complex128)
    for q in range(nkpts):
        a = np.block(
            [[blocks.get(("a", q, ki, kj), z) for kj in range(nkpts)] for ki in range(nkpts)]
        )
        b = np.block(
            [[blocks.get(("b", q, ki, kj), z) for kj in range(nkpts)] for ki in range(nkpts)]
        )
        mat = np.block([[a, b], [-b.conj(), -a.conj()]])
        omega, xy = np.linalg.eigh(mat)
        x, y = xy[:, : np.sum(nov)], xy[:, np.sum(nov) :]


if __name__ == "__main__":
    from pyscf.pbc import gto, scf

    from momentGW.pbc.gw import KGW

    cell = gto.Cell()
    cell.atom = "He 1 1 1; He 3 2 3"
    cell.a = np.eye(3) * 3
    cell.basis = "6-31g"
    cell.verbose = 0
    cell.build()

    mf = scf.KRHF(cell)
    mf.kpts = cell.make_kpts([3, 2, 1])
    mf.with_df = df.GDF(cell, mf.kpts)
    mf.kernel()

    gw = KGW(mf)
    build_se_moments_drpa_exact(gw, 2, *gw.ao2mo(mf.mo_coeff))
