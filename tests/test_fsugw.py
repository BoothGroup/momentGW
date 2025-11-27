"""Tests for `uhf/fsgw.py`."""

import unittest

import numpy as np
from pyscf import dft, gto, lib
from pyscf.agf2 import mpi_helper

from momentGW import fsGW, fsUGW


class Test_fsUGW_vs_fsRGW(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        mol = gto.Mole()
        mol.atom = "Li 0 0 0; H 0 0 1.64"
        mol.basis = "6-31g"
        mol.verbose = 0
        mol.build()

        mf = dft.RKS(mol)
        mf.xc = "hf"
        mf.conv_tol = 1e-12
        mf = mf.density_fit(auxbasis="weigend")
        mf.with_df.build()
        mf.kernel()

        mf.mo_coeff = mpi_helper.bcast_dict(mf.mo_coeff, root=0)
        mf.mo_energy = mpi_helper.bcast_dict(mf.mo_energy, root=0)

        cls.mol, cls.mf = mol, mf

    @classmethod
    def tearDownClass(cls):
        del cls.mol, cls.mf

    def test_dtda(self):
        rgw = fsGW(self.mf)
        rgw.compression = None
        rgw.polarizability = "dtda"
        rgw.kernel(1)

        uhf = self.mf.to_uks()
        uhf.with_df = self.mf.with_df

        ugw = fsUGW(uhf)
        ugw.compression = None
        ugw.polarizability = "dtda"
        ugw.kernel(1)

        self.assertTrue(rgw.converged)
        self.assertTrue(ugw.converged)

        np.testing.assert_allclose(rgw.qp_energy, ugw.qp_energy[0])
        np.testing.assert_allclose(rgw.qp_energy, ugw.qp_energy[1])

    def test_drpa(self):
        rgw = fsGW(self.mf)
        rgw.compression = None
        rgw.polarizability = "drpa"
        rgw.kernel(1)

        uhf = self.mf.to_uks()
        uhf.with_df = self.mf.with_df

        ugw = fsUGW(uhf)
        ugw.compression = None
        ugw.polarizability = "drpa"
        ugw.kernel(1)

        self.assertTrue(rgw.converged)
        self.assertTrue(ugw.converged)

        np.testing.assert_allclose(rgw.qp_energy, ugw.qp_energy[0])
        np.testing.assert_allclose(rgw.qp_energy, ugw.qp_energy[1])

    def test_dtda_compression(self):
        rgw = fsGW(self.mf)
        rgw.compression = "ov,oo"
        rgw.compression_tol = 1e-4
        rgw.polarizability = "dtda"
        rgw.kernel(3)

        uhf = self.mf.to_uks()
        uhf.with_df = self.mf.with_df

        ugw = fsUGW(uhf)
        ugw.compression = "ov,oo"
        ugw.compression_tol = 1e-4
        ugw.polarizability = "dtda"
        ugw.kernel(3)

        self.assertTrue(rgw.converged)
        self.assertTrue(ugw.converged)

        np.testing.assert_allclose(rgw.qp_energy, ugw.qp_energy[0])
        np.testing.assert_allclose(rgw.qp_energy, ugw.qp_energy[1])

    def test_drpa_compression(self):
        rgw = fsGW(self.mf)
        rgw.compression = "ia,oo"
        rgw.compression_tol = 1e-4
        rgw.polarizability = "drpa"
        rgw.kernel(1)

        uhf = self.mf.to_uks()
        uhf.with_df = self.mf.with_df

        ugw = fsUGW(uhf)
        ugw.compression = "ia,oo"
        ugw.compression_tol = 1e-4
        ugw.polarizability = "drpa"
        ugw.kernel(1)

        self.assertTrue(rgw.converged)
        self.assertTrue(ugw.converged)

        np.testing.assert_allclose(rgw.qp_energy, ugw.qp_energy[0], atol=1e-8, rtol=1e-6)
        np.testing.assert_allclose(rgw.qp_energy, ugw.qp_energy[1], atol=1e-8, rtol=1e-6)


class Test_fsUGW(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        mol = gto.Mole()
        mol.atom = "Be 0 0 0; H 0 0 1"
        mol.basis = "sto3g"
        mol.spin = 1
        mol.verbose = 0
        mol.build()

        mf = dft.UKS(mol)
        mf.xc = "hf"
        mf.conv_tol = 1e-12
        mf.kernel()

        mf.mo_coeff = (
            mpi_helper.bcast_dict(mf.mo_coeff[0], root=0),
            mpi_helper.bcast_dict(mf.mo_coeff[1], root=0),
        )
        mf.mo_energy = (
            mpi_helper.bcast_dict(mf.mo_energy[0], root=0),
            mpi_helper.bcast_dict(mf.mo_energy[1], root=0),
        )

        mf = mf.density_fit(auxbasis="cc-pv5z-ri")
        mf.with_df.build()

        cls.mol, cls.mf = mol, mf

    @classmethod
    def tearDownClass(cls):
        del cls.mol, cls.mf

    def test_drpa_regression(self):
        ugw = fsUGW(self.mf)
        ugw.polarizability = "drpa"
        ugw.compression = None
        ugw.npoints = 128
        conv, gf, se, _ = ugw.kernel(nmom_max=3)
        self.assertTrue(conv)
        self.assertAlmostEqual(np.max(ugw.qp_energy[0][self.mf.mo_occ[0] > 0]), -0.2764876796, 6)
        self.assertAlmostEqual(np.max(ugw.qp_energy[1][self.mf.mo_occ[1] > 0]), -0.4765158837, 6)
        self.assertAlmostEqual(np.min(ugw.qp_energy[0][self.mf.mo_occ[0] == 0]), 0.1865688788, 6)
        self.assertAlmostEqual(np.min(ugw.qp_energy[1][self.mf.mo_occ[1] == 0]), 0.2017089856, 6)

    def test_dtda_regression(self):
        ugw = fsUGW(self.mf)
        ugw.polarizability = "dtda"
        ugw.compression = None
        conv, gf, se, _ = ugw.kernel(nmom_max=3)
        self.assertTrue(conv)
        self.assertAlmostEqual(np.max(ugw.qp_energy[0][self.mf.mo_occ[0] > 0]), -0.2755846029, 6)
        self.assertAlmostEqual(np.max(ugw.qp_energy[1][self.mf.mo_occ[1] > 0]), -0.4758147734, 6)
        self.assertAlmostEqual(np.min(ugw.qp_energy[0][self.mf.mo_occ[0] == 0]), 0.1862675078, 6)
        self.assertAlmostEqual(np.min(ugw.qp_energy[1][self.mf.mo_occ[1] == 0]), 0.2013907084, 6)


class Test_fsUGW_no_beta(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        mol = gto.Mole()
        mol.atom = "H 0 0 0; H 0 0 1"
        mol.basis = "sto3g"
        mol.spin = 1
        mol.charge = 1
        mol.verbose = 0
        mol.build()

        mf = dft.UKS(mol)
        mf.xc = "hf"
        mf.conv_tol = 1e-12
        mf.kernel()

        mf.mo_coeff = (
            mpi_helper.bcast_dict(mf.mo_coeff[0], root=0),
            mpi_helper.bcast_dict(mf.mo_coeff[1], root=0),
        )
        mf.mo_energy = (
            mpi_helper.bcast_dict(mf.mo_energy[0], root=0),
            mpi_helper.bcast_dict(mf.mo_energy[1], root=0),
        )

        mf = mf.density_fit(auxbasis="cc-pv5z-ri")
        mf.with_df.build()

        cls.mol, cls.mf = mol, mf

    @classmethod
    def tearDownClass(cls):
        del cls.mol, cls.mf

    def test_dtda_regression(self):
        ugw = fsUGW(self.mf)
        ugw.polarizability = "dtda"
        ugw.compression = None
        ugw.npoints = 128
        conv, gf, se, _ = ugw.kernel(nmom_max=9)
        self.assertAlmostEqual(lib.fp(ugw.qp_energy[0]), -1.1978538038)
        self.assertAlmostEqual(lib.fp(ugw.qp_energy[1]), -0.4667283798)

    def test_drpa_regression(self):
        ugw = fsUGW(self.mf)
        ugw.compression = None
        ugw.npoints = 128
        conv, gf, se, _ = ugw.kernel(nmom_max=9)
        self.assertAlmostEqual(lib.fp(ugw.qp_energy[0]), -1.1984487509)
        self.assertAlmostEqual(lib.fp(ugw.qp_energy[1]), -0.4667656921)


if __name__ == "__main__":
    print("Running tests for fsUGW")
    unittest.main()
