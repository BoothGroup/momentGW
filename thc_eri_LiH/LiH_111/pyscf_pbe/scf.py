#!/usr/bin/env python

import os
from pyscf.pbc import gto, scf, dft, df
from pyscf.gto.basis import parse_nwchem
# import pyscf_interface as pyscf_api

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


# kmf = scf.KRKS(cell, kpts)
# kmf.xc = 'pbe'
# kmf.kernel()
# kmf.analyze()
