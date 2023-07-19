import h5py
import numpy as np

filename = "thc_eri_4.h5"

h5 = h5py.File(filename,'r')

print(list(h5.keys()))

Np = h5['Np']
collocation_matrix = h5['collocation_matrix']
coulomb_matrix = h5['coulomb_matrix']



print(np.max(Np))
print(collocation_matrix)
print(coulomb_matrix)
h5.close()