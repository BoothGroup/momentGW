"""Example of plotting spectral functions for some `GW` methods in `momentGW` with unrestricted
references.
"""

import matplotlib.pyplot as plt
import numpy as np
from pyscf import dft, gto

from momentGW import UGW, evUGW, qsUGW, scUGW

# Define a molecule
mol = gto.Mole()
mol.atom = "Be 0 0 0; H 0 0 1.64"
mol.basis = "cc-pvdz"
mol.spin = 1
mol.verbose = 5
mol.build()

# Run a DFT calculation
mf = dft.UKS(mol)
mf = mf.density_fit()
mf.xc = "hf"
mf.kernel()

# The `gf` and `se` outputs of `momentGW` calculations are `dyson.Lehmann`
# objects, and the functionality in `dyson` can be leveraged for
# postprocessing.

grid = np.linspace(-4, 4, 1024)
eta = 5e-2

# Get a GW spectral function
gw = UGW(mf)
gw.kernel(nmom_max=3)
f_gw = (
    gw.gf[0].on_grid(grid, eta=eta, ordering="advanced"),
    gw.gf[1].on_grid(grid, eta=eta, ordering="advanced"),
)
f_gw = (
    np.trace(f_gw[0], axis1=1, axis2=2).imag / np.pi,
    np.trace(f_gw[1], axis1=1, axis2=2).imag / np.pi,
)

# Get a evGW spectral function
gw = evUGW(mf)
gw.kernel(nmom_max=3)
f_evgw = (
    gw.gf[0].on_grid(grid, eta=eta, ordering="advanced"),
    gw.gf[1].on_grid(grid, eta=eta, ordering="advanced"),
)
f_evgw = (
    np.trace(f_evgw[0], axis1=1, axis2=2).imag / np.pi,
    np.trace(f_evgw[1], axis1=1, axis2=2).imag / np.pi,
)

# Get a scGW spectral function
gw = scUGW(mf)
gw.kernel(nmom_max=3)
f_scgw = (
    gw.gf[0].on_grid(grid, eta=eta, ordering="advanced"),
    gw.gf[1].on_grid(grid, eta=eta, ordering="advanced"),
)
f_scgw = (
    np.trace(f_scgw[0], axis1=1, axis2=2).imag / np.pi,
    np.trace(f_scgw[1], axis1=1, axis2=2).imag / np.pi,
)

# Get a qsGW spectral function
gw = qsUGW(mf)
gw.kernel(nmom_max=3)
f_qsgw = (
    gw.gf[0].on_grid(grid, eta=eta, ordering="advanced"),
    gw.gf[1].on_grid(grid, eta=eta, ordering="advanced"),
)
f_qsgw = (
    np.trace(f_qsgw[0], axis1=1, axis2=2).imag / np.pi,
    np.trace(f_qsgw[1], axis1=1, axis2=2).imag / np.pi,
)

# Get a DFT spectral function
gf = UGW(mf).init_gf()
f_dft = (
    gf[0].on_grid(grid, eta=eta, ordering="advanced"),
    gf[1].on_grid(grid, eta=eta, ordering="advanced"),
)
f_dft = (
    np.trace(f_dft[0], axis1=1, axis2=2).imag / np.pi,
    np.trace(f_dft[1], axis1=1, axis2=2).imag / np.pi,
)

# Plot the spectral functions
fig, axs = plt.subplots(nrows=2, sharex=True)
for s, spin in enumerate(["α", "β"]):
    axs[s].plot(grid, f_dft[s], "k-", label="HF")
    axs[s].plot(grid, f_gw[s], "C0-", label="GW")
    axs[s].plot(grid, f_evgw[s], "C1-", label="evGW")
    axs[s].plot(grid, f_scgw[s], "C2-", label="scGW")
    axs[s].plot(grid, f_qsgw[s], "C3-", label="qsGW")
    axs[s].set_ylabel(f"Spectral function ({spin})")
axs[1].set_xlabel("Energy (Ha)")
axs[0].legend()
plt.tight_layout()
plt.show()
