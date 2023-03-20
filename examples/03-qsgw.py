"""Example of running qsGW calculations.
"""

import numpy as np
from pyscf import gto, dft
from momentGW.qsgw import qsGW
from momentGW.scgw import scGW

mol = gto.M(
        atom="Li 0 0 0; H 0 0 1.64",
        basis="cc-pvdz",
        verbose=5,
)

mf = dft.RKS(mol)
mf = mf.density_fit()
mf.xc = "hf"
mf.kernel()

# qsGW
gw = qsGW(mf)
gw.kernel(nmom_max=3)

# qsGW with a scGW0 self-energy - the solver options can be used to
# control the class used to determine the self-energy used to build
# the static potential
gw = qsGW(mf)
gw.solver = scGW
gw.solver_options = dict(w0=True)
gw.kernel(nmom_max=3)

# qsGW with self-consistent density optimisation of each self-energy
# *before* the quasiparticle equation is solved
gw = qsGW(mf)
gw.solver_options = dict(optimise_chempot=True, fock_loop=True)
gw.kernel(nmom_max=3)
