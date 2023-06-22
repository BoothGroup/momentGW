import matplotlib.pyplot as plt
import numpy as np
import pickle

def LoadData(name_of_pickle: str):
 pickle_file = open(name_of_pickle, 'rb')  # Mode: read + binary
 data_base = pickle.load(pickle_file)
 pickle_file.close()
 return data_base


colours = ['plum', 'mediumorchid', 'darkviolet', 'indigo','darkblue',
           'mediumblue','royalblue','dodgerblue','lightskyblue','thistle']

IP_diff = np.asarray(LoadData('IP_cc_80')).T
EA_diff = np.asarray(LoadData('EA_cc_80')).T

values = np.arange(4,81,4)

plt.figure(figsize=(14,10))

for i in range(5):
    plt.plot(values,IP_diff[i], color=colours[i],label=f'IP {i} energy difference')
    plt.plot(values,EA_diff[i], color=colours[i], linestyle='--', label=f'EA {i} energy difference')

plt.legend()
plt.savefig(fname = 'EA_IP_differences.png',dpi=500)
plt.show()
print(IP_diff.T[4:8])
print(EA_diff.T[4:8])
print(np.min(np.abs(IP_diff)))
print(np.min(np.abs(EA_diff)))


# Gauss = np.asarray(LoadData("mom_zero_gauss_20"))[3::4]
# Gauss_f = np.asarray(LoadData("mom_zero_f_gl_20"))[3::4]
# cc = np.asarray(LoadData("mom_zero_cc_20"))[3::4]
# cc_f = np.asarray(LoadData("mom_zero_f_cc_20"))[3::4]
#
#
# original = np.asarray([1.6261510792155003e-15, 1.1657341758564144e-15])
#
# n_points = np.arange(1,21)
# n_small_points = np.arange(4,21,4)
#
# plt.figure(figsize=(14,10))
# plt.plot(n_points,np.full(20,original[0]), color = 'red', label = 'Original Occ Error')
# plt.plot(n_points,np.full(20,original[1]), color = 'blue', label = 'Original Vir Error')
# plt.plot(n_small_points,Gauss.T[0],color = 'darkred', label = 'Gauss Occ Error', marker = 'x', ms = 10)
# plt.plot(n_small_points,Gauss.T[1], color = 'navy', label = 'Gauss Vir Error', marker = 'x', ms = 10)
# plt.plot(n_small_points,Gauss_f.T[0],color = 'black', label = 'Gauss_f Occ Error', marker = 'x', ms = 10)
# plt.plot(n_small_points,Gauss_f.T[1], color = 'grey', label = 'Gauss_f Vir Error', marker = 'x', ms = 10)
# plt.plot(n_small_points,cc.T[0],color = 'tomato', label = 'CC Occ Error', marker = 'x', ms = 10)
# plt.plot(n_small_points,cc.T[1], color = 'deepskyblue', label = 'CC Vir Error', marker = 'x', ms = 10)
# plt.plot(n_small_points,cc_f.T[0],color = 'pink', label = 'CC Occ Error', marker = 'x', ms = 10)
# plt.plot(n_small_points,cc_f.T[1], color = 'purple', label = 'CC Vir Error', marker = 'x', ms = 10)
# plt.ylabel(r"Error in $\eta^{(0)}$")
# plt.ylabel("Number of points")
# plt.legend()
# plt.show()