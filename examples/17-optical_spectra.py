"""
Example of optical spectra calculations in `momentGW`.
"""

import numpy as np
import matplotlib.pyplot as plt
from pyscf import gto, dft
from momentGW import GW, BSE, dTDA

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

# Run a GW calculation - the optical spectra solver inherits the
# compression scheme used in the GW calculation. Since BSE uses the
# oo and vv slices of the interaction, we include them when
# determining the compression metric.
gw = GW(mf)
gw.compression = "oo,vv,ov"
gw.compression_tol = 1e-8
gw.kernel(nmom_max=5)

# Find the optical spectra using the moments of the dynamic polarizability
# at the level of dTDA. For this we leverage the Bethe--Salpeter equation
# (BSE) solver in `momentGW`, but we do not need to solve the BSE in this
# case.
tda = dTDA(gw, nmom_max=5, integrals=gw.ao2mo())
bse = BSE(gw)
gf = bse.kernel(nmom_max=5, moments=tda.build_dp_moments())
f_tda = gf.on_grid(grid, eta=eta, ordering="advanced")
f_tda = np.trace(f_tda, axis1=1, axis2=2).imag / np.pi

# Find the optical spectra by solver the BSE.
bse = BSE(gw)
gf = bse.kernel(nmom_max=5)
f_bse = gf.on_grid(grid, eta=eta, ordering="advanced")
f_bse = np.trace(f_bse, axis1=1, axis2=2).imag / np.pi

# Plot the optical spectral functions
plt.figure()
plt.plot(grid, f_tda, "C0-", label="dTDA")
plt.plot(grid, f_bse, "C1-", label="BSE")
plt.xlabel("Energy (Ha)")
plt.ylabel("Optical spectral function")
plt.legend()
plt.tight_layout()
plt.show()
