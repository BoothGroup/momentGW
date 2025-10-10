"""Example of customising the Hamiltonian using PySCF for `momentGW`
calculations.
"""

import numpy as np
from pyscf import gto, lib, scf

from momentGW import GW

# Hubbard parameter
n = 10
u = 4.0

# Define a fake molecule
mol = gto.M()
mol.nelectron = 10
mol.verbose = 4

# Define the 1-electron Hamiltonian
h1e = np.zeros((n, n))
for i in range(n - 1):
    h1e[i, i + 1] = h1e[i + 1, i] = -1.0
h1e[0, n - 1] = h1e[n - 1, 0] = -1.0  # Periodic boundary conditions

# Define the 2-electron Hamiltonian
h2e = np.zeros((n, n, n, n))
for i in range(n):
    h2e[i, i, i, i] = u

# Cholesky decomposition of the 2-electron Hamiltonian
# Warning: this is only valid for the Hubbard model (or other models
# with a diagonal 2-electron Hamiltonian)
cderi = np.zeros((n, n, n))
for i in range(n):
    cderi[i, i, i] = np.sqrt(u)
cderi = cderi.reshape(n, n**2)
assert np.allclose(np.dot(cderi.T, cderi), h2e.reshape(n**2, n**2))

# Define a fake mean-field object
mf = scf.RHF(mol).density_fit()
mf.get_hcore = lambda *args: h1e
mf.get_ovlp = lambda *args: np.eye(n)
mf.with_df._cderi = lib.pack_tril(cderi.reshape(n, n, n))
mf.kernel()

# Run a GW calculation
gw = GW(mf)
gw.kernel(nmom_max=5)
