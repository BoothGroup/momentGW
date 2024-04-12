"""
Example of eigenvalue self-consistent GW (evGW) calculations in
`momentGW`.
"""

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

# The eigenvalue self-consistent GW (evGW) method updates the eigenvalues
# used to compute G and/or W, without considering any update to the orbitals.

# Run an evGW calculation
gw = evGW(mf)
gw.conv_tol = 1e-7
gw.conv_tol_moms = 1e-4
gw.kernel(nmom_max=3)

# Use the `g0` parameter to run an evGW0 calculation, where the orbital
# energies of the Green's function in the construction of the self-energy
# are not updated (but W is).
gw = evGW(mf)
gw.g0 = True
gw.conv_tol = 1e-7
gw.conv_tol_moms = 1e-4
gw.kernel(nmom_max=3)

# Use the `w0` parameter to run an evGW0 calculation, where the orbital
# energies of the building of the screened Coulomb interaction are not
# updated.
gw = evGW(mf)
gw.w0 = True
gw.conv_tol = 1e-7
gw.conv_tol_moms = 1e-4
gw.kernel(nmom_max=3)

# If one uses both `g0` and `w0`, then the calculation is equivalent to
# a G0W0 calculation and should converge in a single iteration.
