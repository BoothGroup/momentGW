import numpy as np
from momentGW.gw import GW
from pyscf.pbc import gto, scf, dft
from pyscf import lib, gw
from pyscf.pbc.gw import krgw_cd
#from pyscf.gw import ugw_ac
#from pyscf.gw import urpa

cell = gto.M(
    a = '''0.0, 2.0415, 2.0415
           2.0415, 0.0, 2.0415
           2.0415, 2.0415, 0.0''',
    atom = '''Li  0.      0.      0.
              H 2.0415 2.0415 2.0415''',
    pseudo = 'gth-pbe',
    basis = 'gth-dzvp-molopt-sr',#'gth-szv'
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

#gw = gw.GW(kmf.to_rhf())
#gw.kernel()
#print(gw.mo_energy)

#rpa_obj = urpa.URPA(kmf, frozen=0)
#rpa_obj.kernel()

#gw_obj = ugw_ac.UGWAC(kmf, frozen=0)
#gw_obj.linearized = False
#gw_obj.ac = 'pade'
#gw_obj.kernel()

mgw = GW(kmf)
mgw.kernel(nmom_max=3)
