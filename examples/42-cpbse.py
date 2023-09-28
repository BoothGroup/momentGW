"""Example of running cpBSE calculations.
"""

import numpy as np
import matplotlib.pyplot as plt
from dyson import util
from pyscf import dft, gto, lib

from momentGW import GW, BSE, cpBSE

nmom = 11
ncheb = 100

mol = gto.M(
        atom="Li 0 0 0; H 0 0 1.64",
        basis="cc-pvdz",
        verbose=5,
)

mf = dft.RKS(mol)
mf = mf.density_fit()
mf.xc = "hf"
mf.kernel()

grid = np.linspace(0, 5, 1024)
eta = 1e-1

gw = GW(mf)
gw.polarizability = "dtda"
gw.compression = None
gw.kernel(nmom)

bse = BSE(gw)
integrals = bse.ao2mo()
matvec = bse.build_matvec(integrals)
a = np.array([matvec(v) for v in np.eye(gw.nocc*(gw.nmo-gw.nocc))])
w, v = np.linalg.eigh(a)
with mol.with_common_orig((0, 0, 0)):
    dip = mol.intor_symmetric("int1e_r", comp=3)
ci = integrals.mo_coeff[:, integrals.mo_occ > 0]
ca = integrals.mo_coeff[:, integrals.mo_occ == 0]
dip = lib.einsum("xpq,pi,qa->xia", dip, ci.conj(), ca)
dip = dip.reshape(3, -1)
r = lib.einsum("xp,pi->xi", dip, v)
sf0 = util.build_spectral_function(w, r, grid, eta=eta)

bse.kernel(nmom)
sf1 = util.build_spectral_function(bse.gf.energy, bse.gf.coupling, grid, eta=eta)

emin = np.min(grid)
emax = np.max(grid)
a = (emax - emin) / (2.0 - 1e-3)
b = (emax + emin) / 2.0
scale = (a, b)

cpbse = cpBSE(gw, eta=eta, grid=grid, scale=scale)
sf2 = cpbse.kernel(ncheb)
sf2 = np.trace(cpbse.gf, axis1=1, axis2=2).imag

plt.figure()
plt.plot(grid, sf0, "k-", label="BSE (exact)")
plt.plot(grid, sf1, "C0-", label=f"BSE ({nmom})")
plt.plot(grid, sf2, "C1-", label=f"cpBSE ({ncheb})")
plt.xlabel("Frequency")
plt.ylabel("Spectral function")
plt.legend()
plt.tight_layout()
plt.show()
