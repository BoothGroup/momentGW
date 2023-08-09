"""Example of using THC integrals instead of using Cholesky decomposition.
In this example, the THC integrals are imported from thc_eri_8.h5.
"""

import h5py
import numpy as np
from pyscf import lib
from pyscf.pbc import gto, dft, df
from momentGW.gw import GW
from os.path import abspath, join, dirname

cell = gto.Cell()
cell.atom = "Li 0 0 0; H 2.0415 2.0415 2.0415"
cell.a = (np.ones((3, 3)) - np.eye(3)) * 2.0415
cell.pseudo = "gth-pbe"
cell.basis = "gth-dzvp-molopt-sr"
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
gw.thc_opts = dict(file_path  = abspath(join(dirname(__file__), '..', 'examples/07-thc.h5')))
gw.polarizability = "thc-dtda"
gw.kernel(nmom_max=7)

print("TDA:")
gw = GW(mf)
gw.kernel(nmom_max=7)
