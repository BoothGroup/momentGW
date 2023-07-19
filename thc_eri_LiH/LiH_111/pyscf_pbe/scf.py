from scipy.linalg import cholesky
import os
import h5py



path = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'thc_eri_LiH/LiH_111/thc_eri_8.h5'))
THC_ERI = h5py.File(path,'r')
coll = np.array(THC_ERI['collocation_matrix']).transpose()[0].transpose()
cou = np.array(THC_ERI['coulomb_matrix'][0]).transpose()[0].T
decou = cholesky(cou, lower=True)
THCERI = np.einsum("ip,ap,pq ->qia",coll,coll,decou)
mf.with_df = DummyClass(np.ascontiguousarray(THCERI))