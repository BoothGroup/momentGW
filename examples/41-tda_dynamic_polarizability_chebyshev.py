import numpy as np
import matplotlib.pyplot as plt
from pyscf import gto, scf, gw, lib
from momentGW import GW, TDA
from dyson import CPGF, Lehmann

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

emin = min(w)
emax = max(w)
a = (emax - emin) / (2.0 - 1e-3)
b = (emax + emin) / 2.0

for i, nmom in enumerate([10, 50, 100]):
    m = gf.chebyshev_moment(range(nmom), scaling=(a, b))
    solver = CPGF(m, grid, (a, b), eta=eta)
    s2 = solver.kernel() * np.pi
    plt.plot(grid, s2, f"C{i+1}--", label=f"CPGF ({nmom})")

plt.xlabel("Frequency (Ha)")
plt.ylabel("Dynamic polarizability")
plt.legend()
plt.show()
