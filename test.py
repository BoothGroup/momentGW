
import numpy as np
import pickle
import h5py
from scipy.linalg import cholesky
from os.path import abspath, join, dirname

def LoadData(name_of_pickle:str):
    pickle_file = open(name_of_pickle, 'rb')  # Mode: read + binary
    data_base = pickle.load(pickle_file)
    pickle_file.close()
    return data_base

path = abspath(join(dirname(__file__), '..', 'momentGW/thc_eri_LiH/LiH_111/thc_eri_8.h5'))
THC_ERI = h5py.File(path,'r')
coll = np.array(THC_ERI['collocation_matrix']).transpose()[0].transpose()
cou = np.array(THC_ERI['coulomb_matrix'][0]).transpose()[0].T
decou = cholesky(cou, lower=True)

zeta = LoadData('zeta_THC')
eta = LoadData('eta_CD')
#print(zeta[1])
#print(zeta[3])
print(decou.shape)

zeta = np.einsum('PQ,pQR,RS->pPS',decou.T,zeta,decou)

#print(zeta.shape)
#print(eta.shape)
#print(np.mean(eta[0]-zeta[0]))
#print(eta[3])
#print(zeta[3])
for a in range (7):
     print(np.sum(eta[a]-zeta[a]))

eta_thc = LoadData('pre_mom_fin_THC')
eta_CD = LoadData('pre_mom_fin_CD')
print(np.allclose(eta_thc/2,eta_CD))
print(eta_thc[0][0]/2)
print('')
print(eta_CD[0][0])
