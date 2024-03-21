"""
Example of a script to run UG0W0@dTDA.
"""

from pyscf import dft, gto

from momentGW import UGW

# Define a molecule
mol = gto.Mole()
mol.atom = "Li 0 0 0; H 0 0 1.64"
mol.basis = "cc-pvdz"
mol.verbose = 5
mol.build()

# Run a DFT calculation
mf = dft.UKS(mol)
mf = mf.density_fit()
mf.xc = "hf"
mf.kernel()

# Run a UG0W0@dTDA calculation
gw = UGW(mf)
gw.polarizability = "dTDA"
gw.kernel(nmom_max=1)
