import numpy as np
from momentGW.gw import GW
from pyscf.pbc import gto, scf, dft, gw
from pyscf import lib
#from pyscf.pbc.gw import kgw_slow_supercell

cell = gto.M(
    a = '''0.0, 2.0415, 2.0415
           2.0415, 0.0, 2.0415
           2.0415, 2.0415, 0.0''',
    atom = '''Li  0.      0.      0.
              H 2.0415 2.0415 2.0415''',
    pseudo = 'gth-pbe',
    basis = 'gth-szv',#'gth-dzvp-molopt-sr',
    verbose = 4
)

nk = [1,1,1]
kpts = cell.make_kpts(nk)
cell.exp_to_discard = 0.1
cell.max_memory = 1e10
cell.precision = 1e-6

kmf = scf.KRKS(cell, kpts)
kmf = kmf.rs_density_fit()
kmf.xc = 'pbe'
kmf.kernel()

cderi = list(kmf.with_df.loop())[0]
cderi = lib.unpack_tril(cderi, axis=-1)
kmf.with_df._cderi2 = cderi
#gw = kgw_slow_supercell.GW(kmf, eri = cderi)
#gw.linearized = False
#gw.fc = False
#nocc = gw.nocc
#gw.kernel()
#print(gw.mo_energy)

mgw = GW(kmf)
mgw.kernel(nmom_max=3,ppoints=0)
