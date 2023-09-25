"""
Example of a script to run KUG0W0@dTDA.
"""

import numpy as np
from pyscf.pbc import gto, dft
from momentGW import KUGW

# Define a unit cell
cell = gto.Cell()
cell.atom = "He 0 0 0; He 1 1 1"
cell.a = np.eye(3) * 3
cell.basis = "6-31g"
cell.verbose = 5
cell.build()

# Run a DFT calculation
kpts = cell.make_kpts([3, 1, 1])
mf = dft.KUKS(cell, kpts)
mf = mf.density_fit()
mf.xc = "hf"
mf.kernel()

# Run a KUG0W0@dTDA calculation
gw = KUGW(mf)
gw.polarizability = "dTDA"
gw.kernel(nmom_max=1)
