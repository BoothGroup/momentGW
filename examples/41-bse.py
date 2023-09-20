"""Example of running Bethe-Salpeter equation (BSE) calculations.
"""

import numpy as np
import matplotlib.pyplot as plt
from pyscf import gto, scf, gw, lib
from momentGW import dTDA, BSE, GW
from momentGW.ints import Integrals
from dyson import Lehmann

nmom_max = 13
grid = np.linspace(0, 5, 1024)
eta = 1e-1

mol = gto.M(
    #atom="O 0 0 0; O 0 0 1",
    atom="H 0 0 0; Li 0 0 1.64",
    basis="cc-pvdz",
    verbose=0,
)

mf = scf.RHF(mol)
mf = mf.density_fit()
mf.kernel()

integrals = Integrals(mf.with_df, mf.mo_coeff, mf.mo_occ, store_full=True)
integrals.transform()

gw = GW(mf)
gw.polarizability = "dtda"
gw.kernel(nmom_max)

# Get the exact BSE result for comparison:
with mol.with_common_orig((0, 0, 0)):
    dip = mol.intor_symmetric("int1e_r", comp=3)
    dip = lib.einsum("xpq,pi,qa->xia", dip, mf.mo_coeff[:, mf.mo_occ > 0].conj(), mf.mo_coeff[:, mf.mo_occ == 0])
    dip = dip.reshape(3, -1)
Lia = integrals.Lpq[:, :gw.nocc, gw.nocc:]
Lij = integrals.Lpq[:, :gw.nocc, :gw.nocc]
Lab = integrals.Lpq[:, gw.nocc:, gw.nocc:]
a = np.diag(lib.direct_sum("a-i->ia", gw.qp_energy[mf.mo_occ == 0], gw.qp_energy[mf.mo_occ > 0]).ravel())
a += lib.einsum("Lia,Ljb->iajb", Lia, Lia).reshape(a.shape) * 2.0
a -= lib.einsum("Lab,Lij->iajb", Lab, Lij).reshape(a.shape)
mom = BSE(gw).build_dd_moment_inv(integrals).reshape(gw.nocc, gw.nmo-gw.nocc, gw.nocc, gw.nmo-gw.nocc)
a -= lib.einsum("Lij,Lkc,kcld,Kld,Kab->iajb", Lij, Lia, mom, Lia, Lab).reshape(a.shape)
w, v = np.linalg.eigh(a)
r = lib.einsum("xp,pi->xi", dip, v)
gf = Lehmann(w, r)
s1 = -gf.on_grid(grid, eta=eta, ordering="retarded")

plt.figure()
plt.plot(grid, np.trace(s1, axis1=1, axis2=2).imag, "C0-", label="Exact")

for i, nmom in enumerate(range(1, nmom_max+1, 4)):
    tda = dTDA(gw, nmom, integrals)
    bse = BSE(gw)
    gf = bse.kernel(nmom)
    s2 = -Lehmann(gf.energy, gf.coupling).on_grid(grid, eta=eta, ordering="retarded")
    plt.plot(grid, np.trace(s2, axis1=1, axis2=2).imag, f"C{i+1}-", label=f"MBLGF ({nmom})")

plt.xlabel("Frequency (Ha)")
plt.ylabel("Dynamic polarizability")
plt.legend()
plt.show()
