"""Example of using dTDA screening instead of dRPA.
"""

import numpy as np
from pyscf.pbc import gto, scf
from momentGW.gw import GW
from pyscf import lib
from scipy.linalg import cholesky
import os
import h5py

class DummyClass(object):
    def __init__(self, cderi, kpts, get_jk, cderi2,  *args, **kwargs):
        self._cderi = cderi
        self.kpts = kpts
        self.get_jk = get_jk
        self._cderi2 = cderi2

    def get_naoaux(self):
        return self._cderi.shape[0]

    def kpts(self):
        return self.kpts

    def get_jk(self):
       return self.get_jk

cell = gto.M(
    a = '''0.0, 2.0415, 2.0415
           2.0415, 0.0, 2.0415
           2.0415, 2.0415, 0.0''',
    atom = '''Li  0.      0.      0.
              H 2.0415 2.0415 2.0415''',
    pseudo = 'gth-pbe',
    basis = 'gth-dzvp-molopt-sr',#'gth-szv',
    verbose = 4
)

nk = [1,1,1]
kpts = cell.make_kpts(nk)
cell.exp_to_discard = 0.1
cell.max_memory = 1e10
cell.precision = 1e-6

mf = scf.RKS(cell)
mf = mf.rs_density_fit()
mf.xc = 'pbe'
mf.kernel()

cderi2 = list(mf.with_df.loop())[0]
cderi2 = lib.unpack_tril(cderi2, axis=-1)
# mf.with_df = DummyClass(cderi, mf.with_df.kpts, mf.with_df.get_jk)
path = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'thc_eri_LiH/LiH_111/thc_eri_8.h5'))
THC_ERI = h5py.File(path,'r')
coll = np.array(THC_ERI['collocation_matrix']).transpose()[0].transpose()
cou = np.array(THC_ERI['coulomb_matrix'][0]).transpose()[0].T
decou = cholesky(cou, lower=True)
THCERI = np.einsum("ip,ap,pq ->qia",coll,coll,decou)
mf.with_df = DummyClass(np.ascontiguousarray(THCERI),mf.with_df.kpts, mf.with_df.get_jk, cderi2)


gw = GW(mf)
gw.polarizability = "dtda"
gw.kernel(nmom_max=7)
