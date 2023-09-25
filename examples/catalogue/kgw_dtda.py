"""
Example of a script to run KGW@dTDA.
"""

import numpy as np
from pyscf.pbc import gto, dft
from momentGW import scKGW

# Define a unit cell
cell = gto.Cell()
cell.atom = "He 0 0 0; He 1 1 1"
cell.a = np.eye(3) * 3
cell.basis = "6-31g"
cell.verbose = 5
cell.build()

# Run a DFT calculation
kpts = cell.make_kpts([3, 1, 1])
mf = dft.KRKS(cell, kpts)
mf = mf.density_fit()
mf.xc = "hf"
mf.kernel()

# Run a KGW@dTDA calculation
gw = scKGW(mf)
gw.polarizability = "dTDA"
gw.g0 = True
gw.kernel(nmom_max=1)
