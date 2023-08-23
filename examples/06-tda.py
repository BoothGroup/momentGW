"""Example of using dTDA screening instead of dRPA.
"""
from pyscf import dft, gto

from momentGW.gw import GW

mol = gto.M(
        atom="Li 0 0 0; H 0 0 1.64",
        basis="cc-pvdz",
        verbose=5,
)

mf = dft.RKS(mol)
mf = mf.density_fit()
mf.xc = "hf"
mf.kernel()

gw = GW(mf)
gw.polarizability = "dtda"
gw.kernel(nmom_max=7)
