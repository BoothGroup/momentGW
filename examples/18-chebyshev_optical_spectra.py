"""Example of optical spectra calculations using a Chebyshev polynomial
representation in `momentGW`.
"""

import matplotlib.pyplot as plt
import numpy as np
from pyscf import dft, gto

from momentGW import BSE, GW, cpBSE

# Define a grid
grid = np.linspace(-1.0, 5.0, 1024)
eta = 0.1

# Define a molecule
mol = gto.Mole()
mol.atom = "Li 0 0 0; H 0 0 1.64"
mol.basis = "cc-pvdz"
mol.verbose = 5
mol.build()

# Run a DFT calculation
mf = dft.RKS(mol)
mf = mf.density_fit()
mf.xc = "hf"
mf.kernel()

# Run a GW calculation
gw = GW(mf)
gw.compression = None
gw.kernel(nmom_max=5)

# Run a Bethe--Salpeter equation (BSE) calculation using a monomial
# representation of the moments.
bse = BSE(gw)
gf = bse.kernel(nmom_max=11)
f_bse = gf.on_grid(grid, eta=eta, ordering="advanced")
f_bse = np.trace(f_bse, axis1=1, axis2=2).imag / np.pi

# Run a BSE calculation using a Chebyshev polynomial representation
# of the moments. In this representation the output is already expressed
# on the grid. We must also provide an energy scale of the simulation in
# order to scale the moments in [-1, 1].
a = (grid.max() - grid.min()) / (2.0 - 1e-3)
b = (grid.max() + grid.min()) / 2.0
bse = cpBSE(gw, grid=grid, eta=eta, scale=(a, b))
gf = bse.kernel(nmom_max=100)
f_cpbse = np.trace(gf, axis1=1, axis2=2).imag

# Plot the optical spectral functions
plt.figure()
plt.plot(grid, f_bse, "C0-", label="BSE")
plt.plot(grid, f_cpbse, "C1-", label="cpBSE")
plt.xlabel("Energy (Ha)")
plt.ylabel("Optical spectral function")
plt.legend()
plt.tight_layout()
plt.show()
