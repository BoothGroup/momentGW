"""Example of customising the Hamiltonian using PySCF for `momentGW`
calculations with k-points.
"""

import types

import numpy as np
from pyscf import lib
from pyscf.pbc import df, gto, scf
from pyscf.pbc.lib.kpts_helper import member

from momentGW.pbc import KGW

np.random.seed(12345)

# Parameters
n = 4

# Define a fake molecule
cell = gto.M()
cell.nelectron = n
cell.nao_nr = lambda *args: n
cell.a = np.eye(3)
cell.verbose = 4
kpts = cell.make_kpts([3, 1, 1])

# Define the 1-electron Hamiltonian
h1e = np.einsum("kp,pq->kpq", np.random.random((len(kpts), n)), np.eye(n))

# Define the Cholesky-decomposed 2-electron Hamiltonian
cderi = np.random.random((len(kpts), len(kpts), 10, n, n)).astype(np.complex128) * 0.05
cderi += np.random.random((len(kpts), len(kpts), 10, n, n)) * 0.01j
cderi = cderi + cderi.transpose(1, 0, 2, 4, 3).conj()


# Define a fake density fitting object - note that 2D cells will need
# an additional negative-definite contribution
def sr_loop(self, kpti_kptj=np.zeros((2, 3)), max_memory=2000, compact=True, blksize=None):
    if isinstance(kpti_kptj[0], int):
        ki, kj = kpti_kptj
    else:
        ki = member(kpti_kptj[0], self.kpts)[0]
        kj = member(kpti_kptj[1], self.kpts)[0]
    LpqR = self._cderi[ki, kj].real.copy().reshape(-1, n**2)
    LpqI = self._cderi[ki, kj].imag.copy().reshape(-1, n**2)
    if compact and LpqR.shape[1] == n**2:
        LpqR = lib.pack_tril(LpqR.reshape(-1, n, n))
        LpqI = lib.pack_tril(LpqI.reshape(-1, n, n))
    yield LpqR, LpqI, 1


with_df = df.DF(cell)
with_df.kpts = kpts
with_df._cderi = cderi
with_df.get_naoaux = lambda *args: cderi.shape[2]
with_df.sr_loop = types.MethodType(sr_loop, with_df)

# Define a fake mean-field object
mf = scf.KRHF(cell, kpts).density_fit()
mf.get_hcore = lambda *args: h1e
mf.get_ovlp = lambda *args: np.array([np.eye(n)] * len(kpts))
mf.exxdiv = None
mf.with_df = with_df
mf.kernel()

# Run a GW calculation
gw = KGW(mf)
gw.polarizability = "dTDA"
gw.kernel(nmom_max=3)
