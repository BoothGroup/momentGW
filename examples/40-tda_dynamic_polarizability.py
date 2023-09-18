"""Example of calculating optical spectra at the level of TDA.
"""

import numpy as np
import matplotlib.pyplot as plt
from pyscf import gto, scf, gw, lib
from momentGW import GW, TDA, BSE
from dyson import MBLGF, Lehmann

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

gw = GW(mf)
gw.polarizability = "dtda"
integrals = gw.ao2mo()

# Get the exact TDA result for comparison:
with mol.with_common_orig((0, 0, 0)):
    dip = mol.intor_symmetric("int1e_r", comp=3)
    dip = lib.einsum("xpq,pi,qa->xia", dip, mf.mo_coeff[:, mf.mo_occ > 0].conj(), mf.mo_coeff[:, mf.mo_occ == 0])
    dip = dip.reshape(3, -1)
a = np.diag(lib.direct_sum("a-i->ia", mf.mo_energy[mf.mo_occ == 0], mf.mo_energy[mf.mo_occ > 0]).ravel())
a += lib.einsum("Lx,Ly->xy", integrals.Lia, integrals.Lia) * 2.0
w, v = np.linalg.eigh(a)
r = lib.einsum("xp,pi->xi", dip, v)
gf = Lehmann(w, r)
s1 = -gf.on_grid(grid, eta=eta, ordering="retarded")

plt.figure()
plt.plot(grid, np.trace(s1, axis1=1, axis2=2).imag, "C0-", label="Exact")

for i, nmom in enumerate(range(1, nmom_max+1, 4)):
    # We use the BSE solver but we don't need to solve the BSE,
    # just pass in the dynamic polarizability moments from TDA.
    tda = TDA(gw, nmom, integrals)
    bse = BSE(gw)
    gf = bse.kernel(nmom, moments=tda.build_dp_moments())
    s2 = -Lehmann(gf.energy, gf.coupling).on_grid(grid, eta=eta, ordering="retarded")
    plt.plot(grid, np.trace(s2, axis1=1, axis2=2).imag, f"C{i+1}-", label=f"MBLGF ({nmom})")

plt.xlabel("Frequency (Ha)")
plt.ylabel("Dynamic polarizability")
plt.legend()
plt.show()

