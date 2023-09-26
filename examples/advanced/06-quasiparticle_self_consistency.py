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
# self-consistently via a static approximation to the self-energy
# which is then used to update the orbitals and eigenvalues in an inner
# self-consistent loop.

# A qsGW calculation requires a finite, small spectral broadening (gw.eta)
# to displace the spectral function above or below the real axis in the
# complex plane.
gw = qsGW(mf)
gw.polarizability = "dRPA"
gw.eta = 0.1
gw.kernel(nmom_max=1)

# Run a qsGW calculation using an alternative similarity renormalisation group
# (SRG) regularisation scheme. See arXiv:2303.05984 for more details on
# the SRG scheme. Should give equivalent results for stable eta / srg parameters.
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
# representation. The number of poles scales (linearly) with both system size and
# the `nmom_max` argument. In qsGW, the quasiparticle energies differ from
# the poles of the Green's function. The Green's function in the larger
# space can also be used to determine quasiparticle energies via the states
# which have maximum overlap with G0. These are
# given by `gw.qp_energy`. In qsGW calculations, `gw.qp_energy` stores the
# quasiparticle energies determined by the qsGW loop, while `gw.gf` stores
# the fully correlated Green's function.
print("Poles in correlated Green's function:", gw.gf.naux)
print("Number of quasiparticle energies:", gw.qp_energy.size)
print("G0 energies: ", mf.mo_energy)
print("gw.qp_energy:", gw.qp_energy)
print("gw.gf.as_perturbed_mo_energy():", gw.gf.as_perturbed_mo_energy())
