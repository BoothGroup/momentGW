import numpy as np
import scipy.special
from pyscf import gto, scf, dft, gw, lib
from moment_gw import AGW

mol = gto.M(
        #atom=";".join(["He 0 0 %d" % i for i in range(6)]),
        atom="Li 0 0 0; H 0 0 1.64",
        basis="cc-pvdz",
        verbose=5,
)

mf = dft.RKS(mol)
mf.xc = "hf"
mf.kernel()

exact = gw.GW(mf, freq_int="exact")
exact.kernel()
print(np.max(exact.mo_energy[mf.mo_occ > 0]))

mf = mf.density_fit()

#for n in range(3):
gw = AGW(mf)
gw.diag_sigma = False
conv, gf, se = gw.kernel(nmom=5, vhf_df=True, method="evgw")
gf.remove_uncoupled(tol=1e-8)
print(gf.get_occupied().energy)
