import numpy as np
from pyscf.pbc import gto, scf, dft
from pyscf import lib
from momentGW.gw import GW
import h5py


# cell = gto.M(
#     a = '''0.0, 2.0415, 2.0415
#            2.0415, 0.0, 2.0415
#            2.0415, 2.0415, 0.0''',
#     atom = '''Li  0.      0.      0.
#               H 2.0415 2.0415 2.0415''',
#     pseudo = 'gth-pbe',
#     basis = 'gth-szv',#'gth-dzvp-molopt-sr',
#     verbose = 4
# )
#
# nk = [1,1,1]
# kpts = cell.make_kpts(nk)
# cell.max_memory = 1e10
# cell.precision = 1e-6


cell = gto.M(
    a = '''0.0, 2.0415, 2.0415
           2.0415, 0.0, 2.0415
           2.0415, 2.0415, 0.0''',
    atom = '''Li  0.      0.      0.
              H 2.0415 2.0415 2.0415''',
    pseudo = 'gth-pbe',
    basis = 'gth-dzvp-molopt-sr',
    verbose = 4
)

nk = [1,1,1]
kpts = cell.make_kpts(nk)
cell.max_memory = 1e10
cell.precision = 1e-6


kmf = scf.KRKS(cell, kpts)
kmf.xc = 'pbe'
kmf.kernel()
# kmf.analyze()


# kmf = scf.KRKS(cell, kpts)
# kmf = kmf.rs_density_fit()
# h5py.File('saved_cderi.h5').close()
# kmf.with_df._cderi_to_save = 'saved_cderi.h5'
# kmf.xc = 'pbe'
# kmf.kernel()
print(kmf.get_veff(), kmf.get_j())

cderi = list(kmf.with_df.loop())[0]
cderi = lib.unpack_tril(cderi, axis=-1)
print(cderi.shape)
print(kmf.with_df.get_naoaux(), cell.nao)
# h5py.File('saved_cderi.h5', 'r+').close()
# f = h5py.File('saved_cderi.h5', 'r+')
# f['j3c'] = cderi
# f.close()

cderi_mo = lib.einsum('Qpq,pi,qj->Qij', cderi, kmf.mo_coeff[0], kmf.mo_coeff[0])
print(cderi_mo.shape)
print(cderi.shape)
print(kmf.with_df._cderi)
kmf.mo_occ = np.asarray(kmf.mo_occ)
kmf.with_df._cderi2 = cderi
print(kmf.get_veff(), kmf.get_j())
# kmf.with_df._cderi = 'saved_cderi.h5'
# print(kmf.with_df._cderi)
# print(kmf.with_df.get_naoaux(), cell.nao)





gw = GW(kmf)
gw.kernel(nmom_max=3,ppoints=0)
