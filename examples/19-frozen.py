"""
Example of frozen orbital `momentGW` calculations.
"""

from pyscf import dft, gto

from momentGW import GW, evGW

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

# `frozen` should strictly be a list of orbital indices to freeze, with
# negative indices also supported. Note that PySCF supports other
# formats, but with UHF and/or PBC these notations can become ambiguous.

# Freeze the lowest-energy core orbital
gw = GW(mf)
gw.frozen = [0]
gw.kernel(nmom_max=1)

# Freeze also the two highest-energy unoccupied orbitals, for evGW
evgw = evGW(mf)
evgw.frozen = [0, -2, -1]
evgw.kernel(nmom_max=1)
