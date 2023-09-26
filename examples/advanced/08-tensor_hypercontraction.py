"""
Example of running calculations using tensor hypercontraction (THC) for
the integrals in `momentGW` calculations.
"""

import os
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
mf.xc = "pbe"
mf.kernel()

# Tensor hypercontraction (THC) goes beyond density fitting by further
# decomposing the integrals, and can therefore be used to reduce the
# scaling of the calculation to cubic with system size.

# Currently, a file containing the THC integrals must be provided as
# there is no interface to generate them for ab initio systems yet.
gw = GW(mf)
gw.thc_opts = dict(
    file_path=os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "thc.h5")),
)
gw.polarizability = "THC-dTDA"
gw.kernel(nmom_max=3)
