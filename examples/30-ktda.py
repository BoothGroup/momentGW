"""Example of running k-space GW calculations with dTDA screening.
"""

import numpy as np
from pyscf.pbc import dft, gto

from momentGW.pbc.evgw import evKGW
from momentGW.pbc.gw import KGW
from momentGW.pbc.qsgw import qsKGW
from momentGW.pbc.scgw import scKGW

cell = gto.Cell()
cell.atom = "He 0 0 0; He 1 1 1"
cell.a = np.eye(3) * 3
cell.basis = "6-31g"
cell.verbose = 5
cell.build()

kpts = cell.make_kpts([2, 2, 2])

mf = dft.KRKS(cell, kpts)
mf = mf.density_fit()
mf.xc = "hf"
mf.kernel()

# KGW
gw = KGW(mf)
gw.polarizability = "dtda"
gw.kernel(nmom_max=5)

# qsKGW
gw = qsKGW(mf)
gw.polarizability = "dtda"
gw.srg = 100
gw.kernel(nmom_max=1)

# evKGW
gw = evKGW(mf)
gw.polarizability = "dtda"
gw.kernel(nmom_max=1)

# scKGW
gw = scKGW(mf)
gw.polarizability = "dtda"
gw.kernel(nmom_max=1)
