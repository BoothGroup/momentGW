"""
Example of self-consistent GW (scGW) calculations in `momentGW`.
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

# Full self-consistency updates the energies and orbitals used to compute
# G and W.

# Run an evGW calculation
gw = scGW(mf)
gw.conv_tol = 1e-7
gw.conv_tol_moms = 1e-4
gw.kernel(nmom_max=1)

# Use the `g0` parameter to run an scGW0 calculation, where the orbital
# energies of the Green's function are not updated.
gw = scGW(mf)
gw.g0 = True
gw.conv_tol = 1e-7
gw.conv_tol_moms = 1e-4
gw.kernel(nmom_max=1)

# Use the `w0` parameter to run an scGW0 calculation, where the orbital
# energies of the screened Coulomb interaction are not updated.
gw = scGW(mf)
gw.w0 = True
gw.conv_tol = 1e-7
gw.conv_tol_moms = 1e-4
gw.kernel(nmom_max=1)

# If one uses both `g0` and `w0`, then the calculation is equivalent to
# a G0W0 calculation and should converge in a single iteration.

# The phrase "full self-consistency" may be misleading, as by default the
# solver does not update the static part of the self-energy. This can be
# relaxed according to the new density matrix defined by the updated
# Green's function by setting `fock_loop` to `True`.
gw = scGW(mf)
gw.fock_loop = True
gw.conv_tol = 1e-7
gw.conv_tol_moms = 1e-4
gw.kernel(nmom_max=1)
