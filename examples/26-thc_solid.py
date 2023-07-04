import numpy as np
from momentGW.gw import GW
from pyscf.pbc import gto, scf, dft
from pyscf import lib


cell = gto.M(
    a = '''0.0, 2.0415, 2.0415
           2.0415, 0.0, 2.0415
           2.0415, 2.0415, 0.0''',
    atom = '''Li  0.      0.      0.
              H 2.0415 2.0415 2.0415''',
    pseudo = 'gth-pbe',
    basis = 'gth-szv',#'gth-dzvp-molopt-sr',
    verbose = 1
)

nk = [1,1,1]
kpts = cell.make_kpts(nk)
cell.max_memory = 1e10
cell.precision = 1e-6

kmf = scf.KRKS(cell, kpts)
kmf = kmf.rs_density_fit()
kmf.xc = 'pbe'
kmf.kernel()

cderi = list(kmf.with_df.loop())[0]
cderi = lib.unpack_tril(cderi, axis=-1)
kmf.with_df._cderi2 = cderi


gw = GW(kmf)
gw.kernel(nmom_max=3,ppoints=0)
