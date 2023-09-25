"""
Example of a script to run GW@dRPA.
"""

from pyscf import gto, dft
from momentGW import scGW

# Define a molecule
mol = gto.Mole()
mol.atom = "Li 0 0 0; H 0 0 1.64"
mol.basis = "cc-pvdz"
mol.verbose = 5
mol.build()

# Run a DFT calculation
mf = dft.RKS(mol)
mf = mf.density_fit()
mf.xc = "hf"
mf.kernel()

# Run a GW@dRPA calculation
gw = scGW(mf)
gw.polarizability = "dRPA"
gw.g0 = True
gw.kernel(nmom_max=1)
