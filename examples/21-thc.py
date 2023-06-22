import numpy as np
from pyscf import gto, dft
from momentGW.gw import GW
from vayesta.misc.molecules.molecules import alkane

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

#mol = gto.M(
#        atom="O 0 0 0; O 0 0 1",
#        basis="cc-pvdz",
#        verbose=5,
#)
# mol = gto.M(
#         atom=alkane(5),
#         basis="cc-pvdz",
#         verbose=5,
# )
mol = gto.M(
        atom="Li 0 0 0; H 0 0 1.64",
        basis="cc-pvdz",
        verbose=5,
)

mf = dft.RKS(mol)
mf = mf.density_fit()
mf.xc = "hf"
mf.kernel()
import logging

gw = GW(mf, npoints=48)
IP, EA, errors = gw.kernel(nmom_max=3, ppoints = 48, calc_type='thc')
IP2, EA2, errors2 = gw.kernel(nmom_max=3, ppoints = 4, calc_type='normal')

print(np.flip((IP2-IP)[5:]))
print((EA2-EA)[:5])
#
# num_ppoints = 80
# IP_diff = np.zeros((20, 5))
# EA_diff = np.zeros((20, 5))
# for i in range(4,num_ppoints+1, 4):
#     IP,EA,errors = gw.kernel(nmom_max=3, ppoints = i, calc_type='thc')
#     val = int((i-4)/4)
#     IP_diff[val] = np.flip((IP2-IP)[5:])
#     EA_diff[val] = (EA2 - EA)[:5]
#
# StoreData(IP_diff,'IP_cc_80')
# StoreData(EA_diff,'EA_cc_80')

