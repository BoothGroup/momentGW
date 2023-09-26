"""
Example of the different polarizability models available in `momentGW`
calculations.
"""

from pyscf import gto, dft
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

# Most methods have both dTDA and dRPA available, which are implemented
# efficiently at a cost of O(N^4). The dRPA method is the default for all
# solvers, however only dTDA is available for solid methods in the `pbc`
# module.

# Direct (no exchange) Tamm--Dancoff approximation (dTDA)
gw = GW(mf)
gw.polarizability = "dTDA"
gw.kernel(nmom_max=3)

# Direct (no exchange) random phase approximation (dRPA)
gw = GW(mf)
gw.polarizability = "dRPA"
gw.kernel(nmom_max=3)
