import numpy as np
from pyscf import gto, dft
from momentGW.gw import GW
import matplotlib.pyplot as plt

import pickle


def StoreData(data_list: list, name_of_pickle: str):
    """ Stores list of data. Overwrites any previous data in the pickle file. """
    # Delete previous data
    pickle_file = open(name_of_pickle, 'w+')
    pickle_file.truncate(0)
    pickle_file.close()
    # Write new data
    pickle_file = open(name_of_pickle, 'ab')  # Mode: append + binary
    pickle.dump(data_list, pickle_file)
    pickle_file.close()

mol = gto.M(
        atom="O 0 0 0; O 0 0 1",
        basis="cc-pvdz",
        verbose=5,
)

mf = dft.RKS(mol)
mf = mf.density_fit()
mf.xc = "hf"
mf.kernel()

gw = GW(mf)
# gw.kernel(nmom_max=3, ppoints = 32, calc_type='thc')

num_ppoints = 20
mom_zero_errors = np.zeros((num_ppoints, 2))
for i in range(4,num_ppoints+1, 4):
    mom_zero_errors[i-1] = gw.kernel(nmom_max=3, ppoints = i, calc_type='thc')
print(mom_zero_errors)
StoreData(mom_zero_errors,'mom_zero_f_cc_20')

