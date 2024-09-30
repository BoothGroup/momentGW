"""
Examples of finite size corrections for `momentGW` calculations for periodic solids.
"""

import numpy as np
from pyscf.pbc import dft, gto

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
mf.xc = "b3lyp"
mf.kernel()

# For any `pbc` calculation a finite size correction can be added for
# the divergence associated with G = 0, q=0 for GDF Coulomb integrals.
# This is done by setting the `fsc` attribute to any combination of
# "H", "W", and/or "B" for the Head, Wings and Body portion of the
# correction. The Ewald correction is added in all cases. The default
# is `None` which disables the correction.

# RHF reference
gw = KGW(mf)
gw.polarizability = "dRPA"
gw.fsc = "HWB"
gw.kernel(nmom_max=3)

gw = KGW(mf)
gw.polarizability = "dTDA"
gw.fsc = "HW"
gw.kernel(nmom_max=3)

gw = KGW(mf)
gw.polarizability = "dRPA"
gw.fsc = "H"
gw.kernel(nmom_max=3)
