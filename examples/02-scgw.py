"""Example of running scGW calculations.
"""
from pyscf import dft, gto

from momentGW.scgw import scGW

mol = gto.M(
        atom="Li 0 0 0; H 0 0 1.64",
        basis="cc-pvdz",
        verbose=5,
)

mf = dft.RKS(mol)
mf = mf.density_fit()
mf.xc = "hf"
mf.kernel()

# scGW_0
gw = scGW(mf)
gw.w0 = True
gw.kernel(nmom_max=3)

# scGW with self-consistent density
gw = scGW(mf)
gw.optimise_chempot = True
gw.fock_loop = True
gw.kernel(nmom_max=3)
