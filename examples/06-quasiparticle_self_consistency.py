"""
Example of quasiparticle self-consistent GW (qsGW) calculations in
`momentGW`.
"""

from pyscf import gto, dft
from momentGW import qsGW, evGW

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

# The quasiparticle self-consistent GW (qsGW) method updates
# self-consistently determines a static approximation to the self-energy
# which is then used to update the orbitals and eigenvalues in an inner
# self-consistent loop.

# Run a qsGW calculation with a finite, small spectral broadening to
# displace the spectral function above or below the real axis in the
# complex plane.
gw = qsGW(mf)
gw.polarizability = "dRPA"
gw.eta = 0.1
gw.kernel(nmom_max=1)

# Run a qsGW calculation using the similarity renormalisation group
# (SRG) regularisation scheme. See arXiv:2303.05984 for more details on
# the SRG scheme.
gw = qsGW(mf)
gw.polarizability = "dTDA"
gw.srg = 100
gw.kernel(nmom_max=1)

# One can also use a different solver for obtaining the self-energy within
# the quasiparticle self-consistency loop. Here we use evGW0.
gw = qsGW(mf)
gw.solver = evGW
gw.solver_options = dict(w0=True)
gw.kernel(nmom_max=1)

# In `momentGW`, all GW calculations find a Green's function as a Lehmann
# representation over static poles. The number of poles is controlled by
# the `nmom_max` argument. In qsGW, the quasiparticle energies differ from
# the poles of the Green's function. The Green's function in the larger
# space can also be used to determine quasiparticle energies, which is
# given by `gw.qp_energy`. In qsGW calculations, `gw.qp_energy` stores the
# quasiparticle energies determined by the qsGW loop, while `gw.gf` stores
# the Green's function in the larger space.
# `dyson.Lehmann.as_perturbed_mo_energy()` finds the poles of the Green's
# function that best overlap with the MOs in a fashion similar to that of
# `gw.qp_energy` in the case of non-qsGW calculations.
print("Size of Green's function:", gw.gf.naux)
print("Size of quasiparticle energies:", gw.qp_energy.size)
print("gw.qp_energy:", gw.qp_energy)
print("gw.gf.as_perturbed_mo_energy():", gw.gf.as_perturbed_mo_energy())
