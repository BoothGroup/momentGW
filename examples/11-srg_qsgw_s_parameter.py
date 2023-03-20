"""Replicate Fig. 4 (central panel) of arXiv:2303.05984
"""

import numpy as np
from pyscf import gto, dft, cc
from pyscf.data.nist import HARTREE2EV
from momentGW.qsgw import qsGW
import matplotlib.pyplot as plt

nmom_max = 3

mol = gto.M(
        #atom="Li 0 0 0; H 0 0 1.5949",
        #basis="aug-cc-pvtz",
        atom="O 0 0 0.11779; H 0 0.755453 -0.471161; H 0 0-0.755453 -0.471161",
        basis="cc-pvdz",
        verbose=5,
)

mf = dft.RKS(mol)
mf = mf.density_fit()
mf.xc = "hf"
mf.kernel()

hf = -np.max(mf.mo_energy[mf.mo_occ > 0]) * HARTREE2EV
#ccsd_t = 8.02
#ccsd = cc.CCSD(mf).run().ipccsd(nroots=3)[0]

gw = qsGW(mf)
gw.eta = 0.05
_, gf, se = gw.kernel(nmom_max)
qsgw_eta = -gf.get_occupied().energy.max() * HARTREE2EV

s_params = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0, 500.0, 1000.0][::-1]
qsgw_srg = []

moments = (
        se.get_occupied().moment(range(nmom_max+1)),
        se.get_virtual().moment(range(nmom_max+1)),
)

for s in s_params:
    gw = qsGW(mf)
    gw.srg = s
    gw.diis_space = 10
    gw.conv_tol = 1e-5
    gw.conv_tol_moms = 1
    conv, gf, se = gw.kernel(nmom_max, moments=moments)
    moments = (
            se.get_occupied().moment(range(nmom_max+1)),
            se.get_virtual().moment(range(nmom_max+1)),
    )
    gf.remove_uncoupled(tol=0.8)
    qsgw_srg.append(-gf.get_occupied().energy.max() * HARTREE2EV)

print(qsgw_srg)
qsgw_srg = np.array(qsgw_srg)


plt.figure()
#plt.plot(s_params, [np.abs(hf-ccsd_t)]*len(s_params), "-", color="cyan", label="HF")
#plt.plot(s_params, [np.abs(qsgw_eta-ccsd_t)]*len(s_params), "-", color="blue", label=r"qs$GW$ ($\eta=0.05$)")
#plt.plot(s_params, np.abs(qsgw_srg-ccsd_t), ".-", color="green", label=r"SRG-qs$GW$")
plt.plot(s_params, [hf]*len(s_params), "-", color="cyan", label="HF")
plt.plot(s_params, [qsgw_eta]*len(s_params), "-", color="blue", label=r"qs$GW$ ($\eta=0.05$)")
plt.plot(s_params, qsgw_srg, ".-", color="green", label=r"SRG-qs$GW$")
plt.xlabel(r"$s$")
plt.ylabel("Error in IP (eV)")
plt.xscale("log")
plt.legend()
#plt.savefig("srg_qsgw_s_parameter.pdf", format="pdf")
plt.show()
