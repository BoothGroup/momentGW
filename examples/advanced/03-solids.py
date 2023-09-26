"""
Examples of `momentGW` calculations for periodic solids.
"""

import numpy as np
from pyscf.pbc import gto, dft
from momentGW import KGW, KUGW

# Define a unit cell
cell = gto.Cell()
cell.atom = "He 0 0 0; He 1 1 1"
cell.basis = "6-31g"
cell.a = np.eye(3) * 3
cell.verbose = 5
cell.build()
kpts = cell.make_kpts([3, 1, 1])

# Run a DFT calculation
mf = dft.KRKS(cell, kpts)
mf = mf.density_fit()
mf.xc = "hf"
mf.kernel()

# All core solvers have an additional implementation in the `pbc`
# module implementing the same functionality for k-point sampled
# periodic solids, and the solver can be imported from the `momentGW`
# namespace directly by replacing `GW` with `KGW` in the solver name.

# RHF reference (currently only dTDA screening)
gw = KGW(mf)
gw.polarizability = "dTDA"
gw.kernel(nmom_max=3)

# RHF -> UHF reference
umf = mf.to_uhf()
umf.with_df = mf.with_df
gw = KUGW(umf)
gw.polarizability = "dTDA"
gw.kernel(nmom_max=3)
