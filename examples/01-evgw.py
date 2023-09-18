"""Example of running evGW calculations.
"""

from pyscf import dft, gto

from momentGW.evgw import evGW

mol = gto.M(
        atom="Li 0 0 0; H 0 0 1.64",
        basis="cc-pvdz",
        verbose=5,
)

mf = dft.RKS(mol)
mf = mf.density_fit()
mf.xc = "hf"
mf.kernel()

# evGW
gw = evGW(mf)
gw.kernel(nmom_max=3)

# evGW_0
gw = evGW(mf, w0=True)
gw.kernel(nmom_max=3)
