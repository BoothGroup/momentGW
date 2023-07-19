#!/usr/bin/env python

import os
from pyscf.pbc import gto, scf, dft, df
from pyscf.gto.basis import parse_nwchem
import pyscf_interface as pyscf_api

atom = '''
Li 0.     0.     0.    
H  2.0415 2.0415 2.0415
Li 2.0415 2.0415 0.    
H  4.083  4.083  2.0415
Li 2.0415 0.     2.0415
H  4.083  2.0415 4.083 
Li 4.083  2.0415 2.0415
H  6.1245 4.083  4.083 
Li 0.     2.0415 2.0415
H  2.0415 4.083  4.083 
Li 2.0415 4.083  2.0415
H  4.083  6.1245 4.083 
Li 2.0415 2.0415 4.083 
H  4.083  4.083  6.1245
Li 4.083  4.083  4.083 
H  6.1245 6.1245 6.1245'''

cell = gto.M(
    a = '''0.0, 4.083, 4.083
           4.083, 0.0, 4.083
           4.083, 4.083, 0.0''',
    atom = atom, 
    pseudo = 'gth-pbe',          
    basis = 'gth-dzvp-molopt-sr',
    verbose = 4
)

nk = [1,1,1] 
kpts = cell.make_kpts(nk)

kmf = scf.KRKS(cell, kpts)
kmf.xc = 'pbe'
kmf.kernel()
kmf.analyze()

