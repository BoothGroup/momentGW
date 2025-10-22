"""Examples of the different spin options available in `momentGW` calculations."""

from pyscf import dft, gto

from momentGW import GW, UGW

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

# All core solvers have an additional implementation in the `uhf`
# module implementing the same functionality for unrestricted
# references, and the solver can be imported from the `momentGW`
# namespace directly by replacing `GW` with `UGW` in the solver name.

# RHF reference
gw = GW(mf)
gw.kernel(nmom_max=3)

# RHF -> UHF reference
umf = mf.to_uhf()
umf.with_df = mf.with_df
gw = UGW(umf)
gw.kernel(nmom_max=3)
