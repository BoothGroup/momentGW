"""Example of interpolation of a `momentGW` calculation onto a new k-point mesh."""

import matplotlib.pyplot as plt
import numpy as np
from pyscf.data.nist import HARTREE2EV
from pyscf.pbc import dft, gto

from momentGW.pbc.gw import KGW

# Define a unit cell
cell = gto.Cell()
cell.atom = "H 0 0 0; H 0 0 1"
cell.a = np.array([[25, 0, 0], [0, 25, 0], [0, 0, 2]])
cell.basis = "sto6g"
cell.max_memory = 1e10
cell.verbose = 5
cell.build()

# Define the two k-point meshes
kmesh1 = [1, 1, 3]
kmesh2 = [1, 1, 9]
kpts1 = cell.make_kpts(kmesh1)
kpts2 = cell.make_kpts(kmesh2)

# Run a DFT calculation on the small mesh
mf1 = dft.KRKS(cell, kpts1, xc="hf")
mf1 = mf1.density_fit(auxbasis="weigend")
mf1.exxdiv = None
mf1.conv_tol = 1e-10
mf1.kernel()

# Run a DFT calculation on the large mesh
mf2 = dft.KRKS(cell, kpts2, xc="hf")
mf2 = mf2.density_fit(mf1.with_df.auxbasis)
mf2.exxdiv = mf1.exxdiv
mf2.conv_tol = mf1.conv_tol
mf2.kernel()

# Run a GW calculation on the small mesh
gw1 = KGW(mf1)
gw1.polarizability = "dtda"
gw1.kernel(5)

# Interpolate the GW calculation onto the large mesh
gw2 = gw1.interpolate(mf2, 5)

# Get the quasiparticle energies
e1 = gw1.qp_energy
e2 = gw2.qp_energy


# Plot the quasiparticle energies
def get_xy(kpts, e):
    kpts = kpts.wrap_around(kpts._kpts)[:, 2]
    arg = np.argsort(kpts)
    return kpts[arg], np.array(e)[arg] * HARTREE2EV


plt.figure()
plt.plot(*get_xy(gw1.kpts, e1), "C0.-", label="Original")
plt.plot(*get_xy(gw2.kpts, e2), "C2.-", label="Interpolated")
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.xlabel("k-point")
plt.ylabel("Quasiparticle energy (eV)")
plt.legend(by_label.values(), by_label.keys())
plt.show()
