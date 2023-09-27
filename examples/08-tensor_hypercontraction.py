"""
Example of running calculations using tensor hypercontraction (THC) for
the integrals in `momentGW` calculations.
"""

import os
import numpy as np
from pyscf.pbc import gto, dft
from momentGW import GW

# Define a unit cell
cell = gto.Cell()
cell.atom = """He 0 0 0; He 1 1 1"""
cell.a = np.eye(3) * 3
cell.basis = "6-31g"
cell.verbose = 5
cell.build()
kpts = cell.make_kpts([1, 1, 1])

# Run a Γ-point DFT calculation
mf = dft.RKS(cell)
mf = mf.density_fit()
mf.xc = "pbe"
mf.kernel()

# Tensor hypercontraction (THC) goes beyond density fitting by further
# decomposing the integrals, and can therefore be used to reduce the
# scaling of the calculation to cubic with system size.

# Currently, a file containing the THC integrals must be provided as
# there is no interface to generate them for ab initio systems yet.
gw = GW(mf)
gw.thc_opts = dict(
    file_path=os.path.abspath(os.path.join(os.path.dirname(__file__), "thc.h5")),
)
gw.polarizability = "THC-dTDA"
gw.kernel(nmom_max=3)
