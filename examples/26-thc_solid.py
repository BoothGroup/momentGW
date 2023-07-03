# import numpy as np
from pyscf.pbc import gto, scf, dft
from momentGW.gw import GW

cell = gto.M(
    a = '''0.0, 2.0415, 2.0415
           2.0415, 0.0, 2.0415
           2.0415, 2.0415, 0.0''',
    atom = '''Li  0.      0.      0.
              H 2.0415 2.0415 2.0415''',
    pseudo = 'gth-pbe',
    basis = 'gth-szv',#'gth-dzvp-molopt-sr',
    verbose = 10
)

nk = [1,1,1]
kpts = cell.make_kpts(nk)
cell.max_memory = 1e10
cell.precision = 1e-6

kmf = scf.KRKS(cell, kpts)
kmf = kmf.rs_density_fit()
kmf.xc = 'pbe'
kmf.with_df._cderi_to_save = 'saved_cderi.h5'

kmf.kernel()



#
# cell = gto.M(
#     a = '''0.0, 2.0415, 2.0415
#            2.0415, 0.0, 2.0415
#            2.0415, 2.0415, 0.0''',
#     atom = '''Li  0.      0.      0.
#               H 2.0415 2.0415 2.0415''',
#     pseudo = 'gth-pbe',
#     basis = 'gth-dzvp-molopt-sr',
#     verbose = 4
# )
#
# nk = [1,1,1]
# kpts = cell.make_kpts(nk)
#
#
# kmf = scf.KRKS(cell, kpts)
# kmf.xc = 'pbe'
# kmf.kernel()
# kmf.analyze()

gw = GW(kmf)
gw.kernel(nmom_max=3,ppoints=0)
