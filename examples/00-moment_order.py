"""
Example of the role of moment order in `momentGW` calculations.
"""

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

# `nmom_max` indicates the maximum moment order to be conserved, and
# should always be an odd number.

# The most coarse approximation is to use `nmom_max = 1`.
gw = GW(mf)
gw.kernel(nmom_max=1)

# As `nmom_max` increases, the calculation approaches the limit of a
# traditional GW calculation. Since the moment order indicates a power
# for which a Hamiltonian is raised, large values will become unstable.
gw.kernel(nmom_max=9)
