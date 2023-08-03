import numpy as np
from pyscf.pbc import gto, scf
from momentGW.gw import GW
from pyscf import lib

cell = gto.M(
    a = '''0.0, 2.0415, 2.0415
           2.0415, 0.0, 2.0415
           2.0415, 2.0415, 0.0''',
    atom = '''Li  0.      0.      0.
              H 2.0415 2.0415 2.0415''',
    pseudo = 'gth-pbe',
    basis = 'gth-dzvp-molopt-sr',
    verbose = 4
)

nk = [1,1,1]
kpts = cell.make_kpts(nk)
cell.exp_to_discard = 0.1
cell.max_memory = 1e10
cell.precision = 1e-6

mf = scf.RKS(cell)
mf = mf.rs_density_fit()
mf.xc = 'pbe'
mf.kernel()

gw = GW(mf)
gw.polarizability = "dtda"
gw.ERI = "THCERI"
gw.ERI_file = 'thc_eri_LiH/LiH_111/thc_eri_8.h5'
gw.kernel(nmom_max=7)