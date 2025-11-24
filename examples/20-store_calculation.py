"""Example of how to store a `momentGW` calculations."""

import numpy as np

from pyscf import dft, gto

from momentGW import GW
from momentGW.rpa import dRPA

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

# When running large calculations you may wish to store the key elements of the calculation.
# The key and most expensive terms are the density-density (dd) response moments and the
# self-energy (SE) moments.


nmom_max = 3

# Construct your GW inputs and integrals
gw = GW(mf)
gw.polarizability = "dRPA"
gw.npoints = 24
gw.compression = None
integrals = gw.ao2mo()

# Build the static term
static = gw.build_se_static(integrals)

# If you wish to store the minimum essential information regarding the calculation, the SE-moments
# can just be stored. These moments (or a subset) can then be used to solve Dyson's equation
# Memory cost to store - O(2 * nmom_max * N^2_orb)
# Maximum memory - O(N^3_orb) + O(2 * nmom_max * N^2_orb)

# Build the SE-moments
se_moments = gw.build_se_moments(nmom_max, integrals)

# Solve dyson's equation
gw.kernel(nmom_max, integrals=integrals, moments=se_moments)


# If you also wish to be able to build higher moments to improve the accuracy of your calculation
# in the future, you can also store the dd-moments
# Memory cost to store - O(nmom_max * N^3_orb) + O(2 * nmom_max * N^2_orb)
# Maximum memory - O(nmom_max * N^3_orb) + O(2 * nmom_max * N^2_orb)

# Initialise the screened Coulomb object
rpa = dRPA(gw, nmom_max, integrals)

# Build the DD-moments
moments_dd = rpa.build_dd_moments()

# Build the SE-moments
moments = rpa.build_se_moments(moments_dd)

# Solve dyson's equation
gw.kernel(nmom_max, integrals=integrals, moments=moments)
