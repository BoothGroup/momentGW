"""Examples of running SRG-qsGW calculations.
"""

import numpy as np
from pyscf import gto, dft
from momentGW.qsgw import qsGW

mol = gto.M(
        atom="Li 0 0 0; H 0 0 1.64",
        basis="cc-pvdz",
        verbose=5,
)

mf = dft.RKS(mol)
mf = mf.density_fit()
mf.xc = "hf"
mf.kernel()

# SRG-qsGW with s=1.0
gw = qsGW(mf)
gw.srg = 1.0
gw.kernel(nmom_max=3)
