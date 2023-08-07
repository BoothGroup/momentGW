"""
Tests for `kgw.py`
"""

import unittest

import numpy as np
import pytest
from pyscf.pbc import gto, dft
from pyscf.pbc.tools import k2gamma
from pyscf.agf2 import mpi_helper

from momentGW import GW, KGW


class Test_KGW(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cell = gto.Cell()
        cell.atom = "He 0 0 0; He 1 0 1"
        cell.basis = "6-31g"
        cell.a = np.eye(3) * 3
        cell.precision = 1e-7
        cell.verbose = 0
        cell.build()

        kmesh = [2, 2, 2]
        kpts = cell.make_kpts(kmesh)

        mf = dft.KRKS(cell, kpts, xc="hf")
        mf = mf.density_fit(auxbasis="weigend")
        mf.conv_tol = 1e-10
        mf.kernel()

        for k in range(len(kpts)):
            mf.mo_coeff[k] = mpi_helper.bcast_dict(mf.mo_coeff[k], root=0)
            mf.mo_energy[k] = mpi_helper.bcast_dict(mf.mo_energy[k], root=0)

        smf = k2gamma.k2gamma(mf, kmesh=kmesh)
        smf = smf.density_fit(auxbasis="weigend")

        cls.cell, cls.kpts, cls.mf, cls.smf = cell, kpts, mf, smf

    @classmethod
    def tearDownClass(cls):
        del cls.cell, cls.kpts, cls.mf, cls.smf

    def test_supercell_valid(self):
        # Require real MOs for supercell comparison
        self.assertAlmostEqual(np.max(np.abs(np.array(self.mf.mo_coeff).imag)), 0, 8)

    def _test_vs_supercell(self, gw, kgw, full=False):
        e1 = np.concatenate([gf.energy for gf in kgw.gf])
        w1 = np.concatenate([np.linalg.norm(gf.coupling, axis=0)**2 for gf in kgw.gf])
        mask = np.argsort(e1)
        e1 = e1[mask]
        w1 = w1[mask]
        e2 = gw.gf.energy
        w2 = np.linalg.norm(gw.gf.coupling, axis=0)**2
        if full:
            np.testing.assert_allclose(e1, e2, atol=1e-8)
        else:
            np.testing.assert_allclose(e1[w1 > 1e-1], e2[w2 > 1e-1], atol=1e-8)

    def test_dtda_vs_supercell(self):
        nmom_max = 5

        kgw = KGW(self.mf)
        kgw.polarizability = "dtda"
        kgw.kernel(nmom_max)

        gw = GW(self.smf)
        gw.polarizability = "dtda"
        gw.kernel(nmom_max)

        self._test_vs_supercell(gw, kgw, full=True)

    def test_dtda_vs_supercell_fock_loop(self):
        nmom_max = 5

        kgw = KGW(self.mf)
        kgw.polarizability = "dtda"
        kgw.fock_loop = True
        kgw.kernel(nmom_max)

        gw = GW(self.smf)
        gw.polarizability = "dtda"
        gw.fock_loop = True
        gw.kernel(nmom_max)

        self._test_vs_supercell(gw, kgw)


if __name__ == "__main__":
    print("Running tests for KGW")
    unittest.main()
