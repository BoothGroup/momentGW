"""
Tests for `pbc/uhf/evgw.py`
"""

import unittest

import numpy as np
from pyscf import lib
from pyscf.agf2 import mpi_helper
from pyscf.pbc import dft, gto
from pyscf.pbc.tools import k2gamma

from momentGW import evKGW, evKUGW


class Test_evKUGW_vs_evKGW(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cell = gto.Cell()
        cell.atom = "He 0 0 0; He 1 1 1"
        cell.basis = "6-31g"
        cell.a = np.eye(3) * 3
        cell.max_memory = 1e10
        cell.verbose = 0
        cell.build()

        kmesh = [3, 1, 1]
        kpts = cell.make_kpts(kmesh)

        mf = dft.KRKS(cell, kpts, xc="hf")
        mf = mf.density_fit(auxbasis="weigend")
        mf.with_df._prefer_ccdf = True
        mf.with_df.force_dm_kbuild = True
        mf.exxdiv = None
        mf.conv_tol = 1e-10
        mf.kernel()

        for k in range(len(kpts)):
            mf.mo_coeff[k] = mpi_helper.bcast_dict(mf.mo_coeff[k], root=0)
            mf.mo_energy[k] = mpi_helper.bcast_dict(mf.mo_energy[k], root=0)

        smf = k2gamma.k2gamma(mf, kmesh=kmesh)
        smf = smf.density_fit(auxbasis="weigend")
        smf.exxdiv = None
        smf.with_df._prefer_ccdf = True
        smf.with_df.force_dm_kbuild = True

        cls.cell, cls.kpts, cls.mf, cls.smf = cell, kpts, mf, smf

    @classmethod
    def tearDownClass(cls):
        del cls.cell, cls.kpts, cls.mf, cls.smf

    def test_dtda(self):
        krgw = evKGW(self.mf)
        krgw.compression = None
        krgw.polarizability = "dtda"
        krgw.kernel(1)

        uhf = self.mf.to_uhf()
        uhf.with_df = self.mf.with_df

        kugw = evKUGW(uhf)
        kugw.compression = None
        kugw.polarizability = "dtda"
        kugw.kernel(1)

        self.assertTrue(krgw.converged)
        self.assertTrue(kugw.converged)

        np.testing.assert_allclose(krgw.qp_energy, kugw.qp_energy[0])
        np.testing.assert_allclose(krgw.qp_energy, kugw.qp_energy[1])

    def test_dtda_g0(self):
        krgw = evKGW(self.mf)
        krgw.compression = None
        krgw.polarizability = "dtda"
        krgw.g0 = True
        krgw.kernel(1)

        uhf = self.mf.to_uhf()
        uhf.with_df = self.mf.with_df

        kugw = evKUGW(uhf)
        kugw.compression = None
        kugw.polarizability = "dtda"
        kugw.g0 = True
        kugw.kernel(1)

        self.assertTrue(krgw.converged)
        self.assertTrue(kugw.converged)

        np.testing.assert_allclose(krgw.qp_energy, kugw.qp_energy[0])
        np.testing.assert_allclose(krgw.qp_energy, kugw.qp_energy[1])

    def test_dtda_w0(self):
        krgw = evKGW(self.mf)
        krgw.compression = None
        krgw.polarizability = "dtda"
        krgw.w0 = True
        krgw.kernel(3)

        uhf = self.mf.to_uhf()
        uhf.with_df = self.mf.with_df

        kugw = evKUGW(uhf)
        kugw.compression = None
        kugw.polarizability = "dtda"
        kugw.w0 = True
        kugw.kernel(3)

        self.assertTrue(krgw.converged)
        self.assertTrue(kugw.converged)

        np.testing.assert_allclose(krgw.qp_energy, kugw.qp_energy[0])
        np.testing.assert_allclose(krgw.qp_energy, kugw.qp_energy[1])


class Test_evKUGW_no_beta(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cell = gto.Cell()
        cell.atom = "H 0 0 0; H 1 1 1"
        cell.basis = "sto3g"
        cell.charge = 1
        cell.spin = 1
        cell.a = np.eye(3) * 3
        cell.max_memory = 1e10
        cell.verbose = 0
        cell.build()

        kmesh = [3, 1, 1]
        kpts = cell.make_kpts(kmesh)

        mf = dft.KUKS(cell, kpts, xc="hf")
        mf = mf.density_fit(auxbasis="weigend")
        mf.with_df._prefer_ccdf = True
        mf.with_df.force_dm_kbuild = True
        mf.exxdiv = None
        mf.conv_tol = 1e-10
        mf.kernel()

        for s in range(2):
            for k in range(len(kpts)):
                mf.mo_coeff[s][k] = mpi_helper.bcast_dict(mf.mo_coeff[s][k], root=0)
                mf.mo_energy[s][k] = mpi_helper.bcast_dict(mf.mo_energy[s][k], root=0)

        cls.cell, cls.kpts, cls.mf = cell, kpts, mf

    @classmethod
    def tearDownClass(cls):
        del cls.cell, cls.kpts, cls.mf

    def test_dtda_regression(self):
        kugw = evKUGW(self.mf)
        kugw.compression = None
        kugw.polarizability = "dtda"
        kugw.conv_tol = 1e-7
        kugw.conv_tol_moms = 1e-4
        kugw.kernel(1)

        self.assertTrue(kugw.converged)

        self.assertAlmostEqual(lib.fp(kugw.qp_energy[0]), -0.0131973016)
        self.assertAlmostEqual(lib.fp(kugw.qp_energy[1]), -0.0344673974)


if __name__ == "__main__":
    print("Running tests for evKUGW")
    unittest.main()
