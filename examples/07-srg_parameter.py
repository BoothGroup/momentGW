"""
Example comparing moment-resolved SRG-qsGW with references data from QuAcK.
"""

import matplotlib.pyplot as plt
import numpy as np
from pyscf import dft, gto
from pyscf.data.nist import HARTREE2EV

from momentGW.qsgw import GW, qsGW

nmom_max = 3
which = "ip"

# Define a molecule
mol = gto.Mole()
mol.atom = """O 0 0 -0.06990256; H 0 0.75753241 0.51843495; H 0 -0.75753241 0.51843495"""
mol.basis = "sto-3g"
mol.verbose = 5
mol.build()

# Reference data for this system, obtained using QuAcK: https://github.com/pfloos/QuAcK
data = {
    0.00001: [
        -20.24217557,
        -1.26762769,
        -0.61658973,
        -0.45321139,
        -0.39125373,
        0.60406162,
        0.73951963,
    ],
    0.0001: [
        -20.24169951,
        -1.26761970,
        -0.61661289,
        -0.45320937,
        -0.39123322,
        0.60409582,
        0.73955528,
    ],
    0.001: [
        -20.23791611,
        -1.26759805,
        -0.61683723,
        -0.45313252,
        -0.39093841,
        0.60437892,
        0.73988050,
    ],
    0.01: [
        -20.22324776,
        -1.26755677,
        -0.61906831,
        -0.45241486,
        -0.38784120,
        0.60667163,
        0.74236584,
    ],
    0.1: [-20.13358941, -1.25634134, -0.62531447, -0.44305367, -0.36491306, 0.61200170, 0.74852072],
    1.0: [-20.03125430, -1.21911456, -0.61854347, -0.42168899, -0.33715835, 0.60561426, 0.73843237],
    10.0: [
        -20.02931274,
        -1.19515025,
        -0.61799123,
        -0.42080200,
        -0.33656091,
        0.60545763,
        0.73800488,
    ],
}

# Run a DFT calculation
mf = dft.RKS(mol)
mf = mf.density_fit()
mf.xc = "hf"
mf.kernel()
if which == "ip":
    hf = -np.max(mf.mo_energy[mf.mo_occ > 0]) * HARTREE2EV
else:
    hf = np.min(mf.mo_energy[mf.mo_occ == 0]) * HARTREE2EV

# Run a GW calculation
gw = GW(mf)
_, gf, se, _ = gw.kernel(nmom_max)
if which == "ip":
    gw_eta = -np.max(gw.qp_energy[mf.mo_occ > 0]) * HARTREE2EV
else:
    gw_eta = np.min(gw.qp_energy[mf.mo_occ == 0]) * HARTREE2EV

# Run a qsGW
gw = qsGW(mf)
gw.eta = 0.05
_, gf, se, _ = gw.kernel(nmom_max)
if which == "ip":
    qsgw_eta = -np.max(gw.qp_energy[mf.mo_occ > 0]) * HARTREE2EV
else:
    qsgw_eta = np.min(gw.qp_energy[mf.mo_occ == 0]) * HARTREE2EV

# Run the SRG-qsGW calculations
s_params = sorted(list(data.keys()))[::-1]
qsgw_srg = []

moments = (
    se.occupied().moment(range(nmom_max + 1)),
    se.virtual().moment(range(nmom_max + 1)),
)

for s in s_params:
    gw = qsGW(mf)
    gw.srg = s
    gw.diis_space = 10
    gw.conv_tol = 1e-5
    gw.conv_tol_moms = 1
    conv, gf, se, _ = gw.kernel(nmom_max, moments=moments)
    moments = (
        se.occupied().moment(range(nmom_max + 1)),
        se.virtual().moment(range(nmom_max + 1)),
    )
    if which == "ip":
        qsgw_srg.append(-np.max(gw.qp_energy[mf.mo_occ > 0]) * HARTREE2EV)
    else:
        qsgw_srg.append(np.min(gw.qp_energy[mf.mo_occ == 0]) * HARTREE2EV)

qsgw_srg = np.array(qsgw_srg)

# Reference data
if which == "ip":
    qsgw_srg_ref = -np.array([data[s][mol.nelectron // 2 - 1] for s in s_params]) * HARTREE2EV
else:
    qsgw_srg_ref = np.array([data[s][mol.nelectron // 2] for s in s_params]) * HARTREE2EV

# Plot
plt.figure()
plt.plot(s_params, [hf] * len(s_params), "-", color="C0", label="HF")
plt.plot(
    s_params,
    [gw_eta] * len(s_params),
    "-",
    color="C1",
    label=r"$GW$ ($n_\mathrm{mom}^\mathrm{max}=%d$)" % nmom_max,
)
plt.plot(
    s_params,
    [qsgw_eta] * len(s_params),
    "-",
    color="C2",
    label=r"qs$GW$ ($\eta=0.05$, $n_\mathrm{mom}^\mathrm{max}=%d$)" % nmom_max,
)
plt.plot(
    s_params,
    qsgw_srg,
    ".-",
    color="C3",
    label=r"SRG-qs$GW$ ($n_\mathrm{mom}^\mathrm{max}=%d$)" % nmom_max,
)
plt.plot(s_params, qsgw_srg_ref, ".--", color="C4", label=r"SRG-qs$GW$ (ref)")
plt.xlabel(r"$s$")
plt.ylabel("%s (eV)" % which.upper())
plt.xscale("log")
plt.legend()
plt.show()
