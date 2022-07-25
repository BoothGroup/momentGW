import numpy as np
import scipy.special
from dyson import BlockLanczosSymmSE
from pyscf import gto, scf, dft, gw, lib
from moment_gw import AGW

mol = gto.M(
        atom="Li 0 0 0; H 0 0 1.64",
        basis="cc-pvdz",
        verbose=4,
)

mf = dft.RKS(mol)
mf.xc = "hf"
mf.kernel()

exact = gw.GW(mf, freq_int="exact")
exact.kernel()
print(np.max(exact.mo_energy[mf.mo_occ > 0]))

mf = mf.density_fit()

for n in range(3):
    gw = AGW(mf)
    gw.diag_sigma = True
    conv, gf, se = gw.kernel(nmom=2*n+1)
    gf.remove_uncoupled(tol=1e-8)
    print(gf.get_occupied().energy.max())
