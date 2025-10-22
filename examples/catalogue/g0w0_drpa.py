"""Example of a script to run G0W0@dRPA."""

from pyscf import dft, gto

from momentGW import GW

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

# Run a G0W0@dRPA calculation
gw = GW(mf)
gw.polarizability = "dRPA"
gw.kernel(nmom_max=1)
