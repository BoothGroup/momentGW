"""
Example of plotting spectral functions for some `GW` methods in
`momentGW`.
"""

import matplotlib.pyplot as plt
import numpy as np
from pyscf import dft, gto

from momentGW import GW, evGW, qsGW, scGW, util

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

# The `gf` and `se` outputs of `momentGW` calculations are `dyson.Lehmann`
# objects, and the functionality in `dyson` can be leveraged for
# postprocessing.

grid = np.linspace(-4, 4, 1024)
eta = 5e-2

# Get a GW spectral function
gw = GW(mf)
gw.kernel(nmom_max=3)
f_gw = gw.gf.on_grid(grid, eta=eta, ordering="advanced")
f_gw = np.trace(f_gw, axis1=1, axis2=2).imag / np.pi

# Get a evGW spectral function
gw = evGW(mf)
gw.kernel(nmom_max=3)
f_evgw = gw.gf.on_grid(grid, eta=eta, ordering="advanced")
f_evgw = np.trace(f_evgw, axis1=1, axis2=2).imag / np.pi

# Get a scGW spectral function
gw = scGW(mf)
gw.kernel(nmom_max=3)
f_scgw = gw.gf.on_grid(grid, eta=eta, ordering="advanced")
f_scgw = np.trace(f_scgw, axis1=1, axis2=2).imag / np.pi

# Get a qsGW spectral function
gw = qsGW(mf)
gw.kernel(nmom_max=3)
f_qsgw = gw.gf.on_grid(grid, eta=eta, ordering="advanced")
f_qsgw = np.trace(f_qsgw, axis1=1, axis2=2).imag / np.pi

# Get a DFT spectral function
gf = GW(mf).init_gf()
f_dft = gf.on_grid(grid, eta=eta, ordering="advanced")
f_dft = np.trace(f_dft, axis1=1, axis2=2).imag / np.pi

# Plot the spectral functions
plt.figure()
plt.plot(grid, f_dft, "k-", label="HF")
plt.plot(grid, f_gw, "C0-", label="GW")
plt.plot(grid, f_evgw, "C1-", label="evGW")
plt.plot(grid, f_scgw, "C2-", label="scGW")
plt.plot(grid, f_qsgw, "C3-", label="qsGW")
plt.xlabel("Energy (Ha)")
plt.ylabel("Spectral function")
plt.legend()
plt.tight_layout()
plt.show()
