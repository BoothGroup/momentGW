"""
Example of the RPA numerical integration parameter in `momentGW`
calculations.
"""

from pyscf import gto, dft
from pyscf.data.nist import HARTREE2EV
from momentGW import GW

# Define a molecule
mol = gto.Mole()
mol.atom = "O 0 0 0; O 0 0 1"
mol.basis = "aug-cc-pvdz"
mol.verbose = 0
mol.build()

# Run a DFT calculation
mf = dft.RKS(mol)
mf = mf.density_fit()
mf.xc = "hf"
mf.kernel()

# The numerical integration has a parameter `npoints`, which must be a
# multiple of 4. This controls the number of points used in the
# integrand. Larger values of `nmom_max` tend to require larger values
# of `npoints` to converge.
out = ""
for npoints in [4, 8, 16, 32, 64]:
    gw = GW(mf)
    gw.polarizability = "dRPA"
    gw.npoints = npoints
    gw.kernel(nmom_max=7)
    out += f"npoints = {npoints:#3d}, IP = {gw.qp_energy[mf.mo_occ > 0].max() * HARTREE2EV:#8.8f} eV\n"
print(out)
