"""
Example of using THC integrals instead of using Cholesky decomposition.
In this example, the THC integrals are imported from thc_eri_8.h5.
"""

from os.path import abspath, dirname, join
import numpy as np
from pyscf.pbc import dft, gto

from momentGW.gw import GW

cell = gto.Cell()
cell.atom = """He 0 0 0; He 1 1 1"""
cell.a = np.eye(3) * 3
cell.basis = "6-31g"
cell.verbose = 3
cell.max_memory = 1e10
cell.precision = 1e-6
cell.build()

kpts = cell.make_kpts([1, 1, 1])

# To compare to the Cholesky decomposition code, we need to use
# Gaussian density fitting and not FFT. The THC integrals are
# based of the FFT representation, and so there will be some
# additional error between these calculations.

mf = dft.RKS(cell, xc="pbe")
mf = mf.density_fit()
mf.exxdiv = None
mf.kernel()

print("THC-TDA:")
gw = GW(mf)
gw.thc_opts = dict(file_path=abspath(join(dirname(__file__), "..", "examples/thc.h5")))
gw.polarizability = "thc-dtda"
gw.kernel(nmom_max=7)

print("TDA:")
gw = GW(mf)
gw.kernel(nmom_max=7)
