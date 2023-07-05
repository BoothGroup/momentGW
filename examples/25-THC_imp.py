import numpy as np
from momentGW.gw import GW
from pyscf.pbc import gto, scf, dft, gw
from pyscf import lib
import h5py
from scipy.linalg import cholesky
import os


cell = gto.M(
    a = '''0.0, 2.0415, 2.0415
           2.0415, 0.0, 2.0415
           2.0415, 2.0415, 0.0''',
    atom = '''Li  0.      0.      0.
              H 2.0415 2.0415 2.0415''',
    pseudo = 'gth-pbe',
    basis = 'gth-szv',#'gth-dzvp-molopt-sr',
    verbose = 4
)

nk = [1,1,1]
kpts = cell.make_kpts(nk)
cell.exp_to_discard = 0.1
cell.max_memory = 1e10
cell.precision = 1e-6

kmf = scf.KRKS(cell, kpts)
kmf = kmf.rs_density_fit()
kmf.xc = 'pbe'
kmf.kernel()
path = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'thc_eri_LiH/LiH_111/thc_eri_2.h5'))

THC_ERI = h5py.File(path,'r')
collocation_matrix = np.array(THC_ERI['collocation_matrix']).transpose()[0].transpose()
coulomb_matrix = np.array(THC_ERI['coulomb_matrix'][0]).transpose()[0]
decomp_coulomb = cholesky(coulomb_matrix, lower=True)
cderi = lib.einsum("np,mp,pq->nmq",collocation_matrix,collocation_matrix,decomp_coulomb)

cderi2 = list(kmf.with_df.loop())[0]
cderi2 = lib.unpack_tril(cderi2, axis=-1)
np.allclose(cderi,cderi2)

kmf.with_df._cderi2 = cderi

# mgw = GW(kmf)
# mgw.kernel(nmom_max=3)
