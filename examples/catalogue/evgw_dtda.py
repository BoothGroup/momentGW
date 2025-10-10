"""Example of a script to run evGW@dTDA."""

from pyscf import dft, gto

from momentGW import evGW

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

# Run a evGW@dTDA calculation
gw = evGW(mf)
gw.polarizability = "dTDA"
gw.g0 = True
gw.kernel(nmom_max=1)
