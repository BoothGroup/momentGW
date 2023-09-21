"""Example of running cpBSE calculations.
"""

import numpy as np
import matplotlib.pyplot as plt
from dyson import util
from pyscf import dft, gto, lib

from momentGW import GW, BSE, cpBSE

nmom = 5
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
bse.kernel(nmom)
sf1 = util.build_spectral_function(bse.gf.energy, bse.gf.coupling, grid, eta=eta)

emin = 0.0
emax = np.max(lib.direct_sum("a-i->ia", mf.mo_energy[mf.mo_occ == 0], mf.mo_energy[mf.mo_occ > 0]))
a = (emax - emin) / (2.0 - 1e-3)
b = (emax + emin) / 2.0
scale = (a, b)

cpbse = cpBSE(gw, eta=eta, grid=grid, scale=scale)
cpbse.kernel(ncheb)
sf2 = np.trace(cpbse.gf, axis1=1, axis2=2).imag / np.pi

plt.figure()
plt.plot(grid, sf1, "C0-", label="BSE")
plt.plot(grid, sf2, "C1-", label="cpBSE")
plt.xlabel("Frequency")
plt.ylabel("Spectral function")
plt.legend()
plt.tight_layout()
plt.show()
