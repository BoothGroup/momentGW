import numpy as np
import matplotlib.pyplot as plt
from pyscf import gto, dft, gw, lib
from moment_gw import AGW

nmom_max = 8
ip = True

mol = gto.M(
        #atom="Li 0 0 0; H 0 0 1.64",
        #atom="O 0 0 0; H 0 0 1; H 0 1 0",
        atom="O 0 0 0; O 0 0 1",
        basis="cc-pvdz",
        verbose=5,
)

mf = dft.RKS(mol)
mf.xc = "hf"
mf.kernel()

exact = gw.GW(mf, freq_int="exact")
exact.kernel()

ac = gw.GW(mf, freq_int="ac")
ac.kernel()

#mf = mf.density_fit(auxbasis="cc-pv5z-ri")
mf = mf.density_fit()

plt.figure()

plt.style.use('seaborn-talk')
plt.rc('axes', facecolor='whitesmoke')
plt.rc('figure', facecolor='white')
plt.rc('lines', markeredgecolor='k', markeredgewidth=1.0)

if ip:
    plt.plot(list(range(nmom_max+1)), [exact.mo_energy[exact.mo_occ > 0].max()]*(nmom_max+1), "k-", label="Exact")
    plt.plot(list(range(nmom_max+1)), [ac.mo_energy[ac.mo_occ > 0].max()]*(nmom_max+1), "k--", label="AC")
else:
    plt.plot(list(range(nmom_max+1)), [exact.mo_energy[exact.mo_occ == 0].min()]*(nmom_max+1), "k-", label="Exact")
    plt.plot(list(range(nmom_max+1)), [ac.mo_energy[ac.mo_occ == 0].min()]*(nmom_max+1), "k--", label="AC")

ips = []
eas = []
for n in range(nmom_max+1):
    gw = AGW(mf)
    gw.diag_sigma = True
    gw.optimise_chempot = False
    conv, gf, se = gw.kernel(nmom=2*n+1, vhf_df=True)
    gf.remove_uncoupled(tol=1e-3)
    ips.append(gf.get_occupied().energy.max())
    eas.append(gf.get_virtual().energy.min())

if ip:
    plt.plot(list(range(nmom_max+1)), ips, "C0.-", label="Diagonal")
else:
    plt.plot(list(range(nmom_max+1)), eas, "C0.-", label="Diagonal")

#ips = []
#eas = []
#for n in range(nmom_max+1):
#    gw = AGW(mf)
#    gw.diag_sigma = True
#    gw.optimise_chempot = True
#    conv, gf, se = gw.kernel(nmom=2*n+1, vhf_df=True)
#    gf.remove_uncoupled(tol=1e-3)
#    ips.append(gf.get_occupied().energy.max())
#    eas.append(gf.get_virtual().energy.min())
#
#if ip:
#    plt.plot(list(range(nmom_max+1)), ips, "C1.-", label="Diagonal, SE shift")
#else:
#    plt.plot(list(range(nmom_max+1)), eas, "C1.-", label="Diagonal, SE shift")

ips = []
eas = []
for n in range(nmom_max+1):
    gw = AGW(mf)
    gw.diag_sigma = False
    gw.optimise_chempot = False
    conv, gf, se = gw.kernel(nmom=2*n+1, vhf_df=True)
    gf.remove_uncoupled(tol=1e-3)
    ips.append(gf.get_occupied().energy.max())
    eas.append(gf.get_virtual().energy.min())

if ip:
    plt.plot(list(range(nmom_max+1)), ips, "C1.-", label="Non-diagonal")
else:
    plt.plot(list(range(nmom_max+1)), eas, "C1.-", label="Non-diagonal")

#ips = []
#eas = []
#for n in range(nmom_max+1):
#    gw = AGW(mf)
#    gw.diag_sigma = False
#    gw.optimise_chempot = True
#    conv, gf, se = gw.kernel(nmom=2*n+1, vhf_df=True)
#    gf.remove_uncoupled(tol=1e-3)
#    ips.append(gf.get_occupied().energy.max())
#    eas.append(gf.get_virtual().energy.min())
#
#if ip:
#    plt.plot(list(range(nmom_max+1)), ips, "C3.-", label="Non-diagonal, SE shift")
#else:
#    plt.plot(list(range(nmom_max+1)), eas, "C3.-", label="Non-diagonal, SE shift")

plt.xlabel("Number of iterations")
plt.ylabel(r"Root")

plt.legend()
plt.tight_layout()

plt.savefig("convergence.png", dpi=128)
#plt.show()
