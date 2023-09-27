"""
Example of extracting properties from `momentGW` calculations.
"""

import numpy as np
from pyscf import gto, dft
from pyscf.data.nist import HARTREE2EV
from momentGW import GW

np.set_printoptions(edgeitems=1000, linewidth=1000, precision=2)

# Define a molecule
mol = gto.Mole()
mol.atom = "Li 0 0 0; H 0 0 1.64"
mol.basis = "6-31g"
mol.verbose = 0
mol.build()

# Run a DFT calculation
mf = dft.RKS(mol)
mf = mf.density_fit()
mf.xc = "hf"
mf.kernel()

# Run a GW calculation
gw = GW(mf)
gw.kernel(nmom_max=3)

# Extract IPs, EAs, and band gaps:

# Method 1) using the `gw.qp_energy` attribute:
ip = -gw.qp_energy[mf.mo_occ > 0].max() * HARTREE2EV
ea = gw.qp_energy[mf.mo_occ == 0].min() * HARTREE2EV
print("\nExcitations from `gw.qp_energy`:")
print("IP : %7.4f eV" % ip)
print("EA : %7.4f eV" % ea)
print("Gap: %7.4f eV" % (ea - ip))

# Method 2) using the `gw.gf` attribute:
ip = -gw.gf.occupied().energies[-1] * HARTREE2EV
nip = gw.gf.occupied().weights()[-1]
ea = gw.gf.virtual().energies[0] * HARTREE2EV
nea = gw.gf.virtual().weights()[0]
print("\nExcitations from `gw.gf`:")
print("IP : %7.4f eV  (weight: %.2g)" % (ip, nip))
print("EA : %7.4f eV  (weight: %.2g)" % (ea, nea))
print("Gap: %7.4f eV" % (ea - ip))

# For larger examples, especially with full self-consistency and larger
# `nmom_max`, there may be low-weighted poles inside the band gap in the
# `gw.gf` object. These can be removed by either setting `gw.weight_tol`
# to a threshold which removes these poles between each iteration in the
# self-consistency, or by using `gw.gf.physical().occupied()` and
# `gw.gf.physical().virtual()` when finding the IPs and EAs.


# A number of energies are available, and their exact form depends on
# which Green's function is used.

# 1) One-body energy, using either the Hartree--Fock Green's function or
#    the correlated GW Green's function:
print("\nOne-body energies:")
print("E(1b, G0): %8.4f eV" % ((gw.energy_hf(gf=gw.init_gf()) + gw.energy_nuc()) * HARTREE2EV))
print("E(1b, G) : %8.4f eV" % ((gw.energy_hf(gf=gw.gf) + gw.energy_nuc()) * HARTREE2EV))

# 2) Two-body energy via the Galitskii--Migdal formula, using either the
#    Hartree--Fock Green's function or the correlated GW Green's function:
print("\nTwo-body energies:")
print("E(2b, G0): %8.4f eV" % (gw.energy_gm(g0=True) * HARTREE2EV))
print("E(2b, G) : %8.4f eV" % (gw.energy_gm(g0=False) * HARTREE2EV))


# First-order density matrices are available using the correlated GW
# Green's function. For the correlated GW Green's function, the trace
# will not be exactly the particle number, as one-shot GW is not a
# conserving approximation. This can be relaxed using self-consistency.

# 1) First-order density matrix using the Hartree--Fock Green's function:
print("\nRDM1 with Hartree--Fock Green's function:")
print(gw.make_rdm1(gf=gw.init_gf()))
print("Tr[D]: %.4f" % gw.make_rdm1(gf=gw.init_gf()).trace())

# 2) First-order density matrix using the correlated GW Green's function:
print("\nRDM1 with GW Green's function:")
print(gw.make_rdm1(gf=gw.gf))
print("Tr[D]: %.4f" % gw.make_rdm1(gf=gw.gf).trace())
