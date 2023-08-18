"""Example of interpolation of a GW calculation onto a new k-point mesh.
"""

import numpy as np
import matplotlib.pyplot as plt
from pyscf.pbc import gto, dft
from momentGW.pbc.gw import KGW

cell = gto.Cell()
cell.atom = "H 0 0 0; H 0 0 1"
cell.a = np.array([[25, 0, 0], [0, 25, 0], [0, 0, 2]])
cell.basis = "sto6g"
cell.max_memory = 1e10
cell.verbose = 5
cell.build()

kmesh1 = [1, 1, 3]
kmesh2 = [1, 1, 9]
kpts1 = cell.make_kpts(kmesh1)
kpts2 = cell.make_kpts(kmesh2)

mf1 = dft.KRKS(cell, kpts1, xc="hf")
mf1 = mf1.density_fit(auxbasis="weigend")
mf1.exxdiv = None
mf1.conv_tol = 1e-10
mf1.kernel()

mf2 = dft.KRKS(cell, kpts2, xc="hf")
mf2 = mf2.density_fit(mf1.with_df.auxbasis)
mf2.exxdiv = mf1.exxdiv
mf2.conv_tol = mf1.conv_tol
mf2.kernel()

gw1 = KGW(mf1)
gw1.polarizability = "dtda"
gw1.kernel(5)

gw2 = gw1.interpolate(mf2, 5)

e1 = gw1.qp_energy
e2 = gw2.qp_energy


def get_xy(kpts, e):
    kpts = kpts.wrap_around(kpts._kpts)[:, 2]
    arg = np.argsort(kpts)
    return kpts[arg], np.array(e)[arg]

plt.figure()
plt.plot(*get_xy(gw1.kpts, e1), "C0.-", label="original")
plt.plot(*get_xy(gw2.kpts, e2), "C2.-", label="interpolated")
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys())
plt.show()
