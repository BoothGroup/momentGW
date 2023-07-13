import numpy as np
import matplotlib.pyplot as plt
from pyscf import gto, dft
from pyscf.data.nist import HARTREE2EV
from momentGW import GW, scGW, evGW, qsGW

mol = gto.Mole()
mol.atom = "Li 0 0 0; H 0 0 1.64"
mol.basis = "cc-pvdz"
mol.verbose = 0
mol.build()

mf = dft.RKS(mol, xc="hf")
mf = mf.density_fit()
mf.kernel()

gaps = {
    "g0w0": [],
    "gw0": [],
    "gw": [],
    "evgw0": [],
    "evgw": [],
    "qsgw": [],
}

def get_gap(gw):
    if isinstance(gw, qsGW):
        ip = -gw.gf.energy[mol.nelectron//2-1]
        ea = gw.gf.energy[mol.nelectron//2]
    else:
        ip = -gw.gf.energy[np.argmax(gw.gf.coupling[mol.nelectron//2-1]**2)]
        ea = gw.gf.energy[np.argmax(gw.gf.coupling[mol.nelectron//2]**2)]
    gap = ip + ea
    return gap * HARTREE2EV

for nmom_max in [1, 3, 5, 7, 9, 11]:
    print("g0w0", nmom_max)
    gw = GW(mf)
    gw.kernel(nmom_max)
    gaps["g0w0"].append(get_gap(gw))

for nmom_max in [1, 3, 5, 7, 9]:
    print("gw0", nmom_max)
    gw = scGW(mf, w0=True)
    gw.kernel(nmom_max)
    gaps["gw0"].append(get_gap(gw))

for nmom_max in [1, 3, 5, 7, 9]:
    print("gw", nmom_max)
    gw = scGW(mf)
    gw.kernel(nmom_max)
    gaps["gw"].append(get_gap(gw))

for nmom_max in [1, 3, 5, 7, 9]:
    print("evgw0", nmom_max)
    gw = evGW(mf, w0=True)
    gw.kernel(nmom_max)
    gaps["evgw0"].append(get_gap(gw))

for nmom_max in [1, 3, 5, 7, 9]:
    print("evgw", nmom_max)
    gw = evGW(mf)
    gw.kernel(nmom_max)
    gaps["evgw"].append(get_gap(gw))

for nmom_max in [1, 3, 5, 7, 8]:
    print("qsgw", nmom_max)
    gw = qsGW(mf)
    gw.kernel(nmom_max)
    gaps["qsgw"].append(get_gap(gw))

plt.figure()
for i, key in enumerate(["g0w0", "gw0", "gw", "evgw0", "evgw", "qsgw"]):
    plt.plot(np.linspace(-0.25, 0.25, num=len(gaps[key]), endpoint=True)+i, gaps[key], "C%d.-"%i)
plt.xticks(range(6), [r"$G_0W_0$", r"$GW_0$", r"$GW$", r"ev$GW_0$", r"ev$GW$", r"qs$GW$"])
plt.ylabel("Gap (eV)")
plt.tight_layout()
plt.show()
