"""Example of the compression parameter in `momentGW` calculations."""

from pyscf import dft, gto
from pyscf.data.nist import HARTREE2EV

from momentGW import GW, scGW

# Define a molecule
mol = gto.Mole()
mol.atom = "Li 0 0 0; H 0 0 1.64"
mol.basis = "cc-pvdz"
mol.verbose = 0
mol.build()

# Run a DFT calculation
mf = dft.RKS(mol)
mf = mf.density_fit()
mf.xc = "hf"
mf.kernel()

# The parameter `compression_tol` controls the threshold for eigenvalues
# in the inner product of the interaction.
out = ""
for compression_tol in [1e-2, 1e-4, 1e-6, 1e-8, 1e-10]:
    gw = GW(mf)
    gw.polarizability = "dRPA"
    gw.compression_tol = compression_tol
    gw.kernel(nmom_max=5)
    out += f"compression_tol = {compression_tol:#7.1g}, IP = {gw.qp_energy[mf.mo_occ > 0].max() * HARTREE2EV:#8.8f} eV\n"

# The parameter `compression` controls the blocks of the interaction that
# are used to construct the inner product.
out += "\n"
for compression in ["oo", "ov", "ov,oo", "ov,oo,vv", None]:
    gw = GW(mf)
    gw.polarizability = "dRPA"
    gw.compression = compression
    gw.compression_tol = 1e-6
    gw.kernel(nmom_max=5)
    out += (
        f"compression = %8s, IP = {gw.qp_energy[mf.mo_occ > 0].max() * HARTREE2EV:#8.8f} eV\n"
        % compression
    )

# For self-consistent GW, one can also use `compression="ia"` which is
# equivalent to `compression="ov"`, but updates the compression metric
# between iterations.
out += "\n"
for compression in ["ov", "ia", None]:
    gw = scGW(mf)
    gw.polarizability = "dTDA"
    gw.compression = compression
    gw.compression_tol = 1e-4
    gw.kernel(nmom_max=1)
    out += (
        f"compression = %8s, IP = {gw.qp_energy[mf.mo_occ > 0].max() * HARTREE2EV:#8.8f} eV\n"
        % compression
    )

print(out)
