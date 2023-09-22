"""
Tests for `uhf/scgw.py`.
"""

import unittest

import numpy as np
import pytest
from pyscf import dft, gto, gw, lib, tdscf
from pyscf.agf2 import mpi_helper

from momentGW import scGW
from momentGW.uhf import scUGW


class Test_scUGW_vs_scRGW(unittest.TestCase):
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
        mf = mf.density_fit()
        mf.with_df.build()
        mf.kernel()

        mf.mo_coeff = mpi_helper.bcast_dict(mf.mo_coeff, root=0)
        mf.mo_energy = mpi_helper.bcast_dict(mf.mo_energy, root=0)

        cls.mol, cls.mf = mol, mf

    @classmethod
    def tearDownClass(cls):
        del cls.mol, cls.mf

    def test_dtda(self):
        rgw = scGW(self.mf)
        rgw.compression = None
        rgw.polarizability = "dtda"
        rgw.max_cycle = 250
        rgw.conv_tol_moms = 1e-4
        rgw.conv_tol = 1e-8
        rgw.kernel(5)

        uhf = self.mf.to_uks()
        uhf.with_df = self.mf.with_df

        ugw = scUGW(uhf)
        ugw.compression = None
        ugw.polarizability = "dtda"
        ugw.max_cycle = 250
        ugw.conv_tol_moms = 1e-4
        ugw.conv_tol = 1e-8
        ugw.kernel(5)

        self.assertTrue(rgw.converged)
        self.assertTrue(ugw.converged)

        np.testing.assert_allclose(rgw.qp_energy, ugw.qp_energy[0], atol=1e-4, rtol=1e-4)
        np.testing.assert_allclose(rgw.qp_energy, ugw.qp_energy[1], atol=1e-4, rtol=1e-4)

    def test_drpa_g0(self):
        rgw = scGW(self.mf)
        rgw.compression = None
        rgw.polarizability = "drpa"
        rgw.g0 = True
        rgw.max_cycle = 250
        rgw.conv_tol_moms = 1e-4
        rgw.conv_tol = 1e-8
        rgw.kernel(3)

        uhf = self.mf.to_uks()
        uhf.with_df = self.mf.with_df

        ugw = scUGW(uhf)
        ugw.compression = None
        ugw.polarizability = "drpa"
        ugw.g0 = True
        ugw.max_cycle = 250
        ugw.conv_tol_moms = 1e-4
        ugw.conv_tol = 1e-8
        ugw.kernel(3)

        self.assertTrue(rgw.converged)
        self.assertTrue(ugw.converged)

        np.testing.assert_allclose(rgw.qp_energy, ugw.qp_energy[0], atol=1e-4, rtol=1e-4)
        np.testing.assert_allclose(rgw.qp_energy, ugw.qp_energy[1], atol=1e-4, rtol=1e-4)

    def test_dtda_w0_compression(self):
        rgw = scGW(self.mf)
        rgw.compression = "ov,oo"
        rgw.compression_tol = 1e-4
        rgw.polarizability = "dtda"
        rgw.w0 = True
        rgw.max_cycle = 250
        rgw.conv_tol_moms = 1e-4
        rgw.conv_tol = 1e-8
        rgw.kernel(5)

        uhf = self.mf.to_uks()
        uhf.with_df = self.mf.with_df

        ugw = scUGW(uhf)
        ugw.compression = "ov,oo"
        ugw.compression_tol = 1e-4
        ugw.polarizability = "dtda"
        ugw.w0 = True
        ugw.max_cycle = 250
        ugw.conv_tol_moms = 1e-4
        ugw.conv_tol = 1e-8
        ugw.kernel(5)

        self.assertTrue(rgw.converged)
        self.assertTrue(ugw.converged)

        np.testing.assert_allclose(rgw.qp_energy, ugw.qp_energy[0], atol=1e-4, rtol=1e-4)
        np.testing.assert_allclose(rgw.qp_energy, ugw.qp_energy[1], atol=1e-4, rtol=1e-4)

    def test_drpa_compression_no_fock(self):
        rgw = scGW(self.mf)
        rgw.compression = "ia,oo"
        rgw.compression_tol = 1e-4
        rgw.polarizability = "drpa"
        rgw.max_cycle = 250
        rgw.conv_tol_moms = 1e-4
        rgw.conv_tol = 1e-8
        rgw.fock_loop = False
        rgw.kernel(1)

        uhf = self.mf.to_uks()
        uhf.with_df = self.mf.with_df

        ugw = scUGW(uhf)
        ugw.compression = "ia,oo"
        ugw.compression_tol = 1e-4
        ugw.polarizability = "drpa"
        ugw.max_cycle = 250
        ugw.conv_tol_moms = 1e-4
        ugw.conv_tol = 1e-8
        ugw.fock_loop = False
        ugw.kernel(1)

        self.assertTrue(rgw.converged)
        self.assertTrue(ugw.converged)

        np.testing.assert_allclose(rgw.qp_energy, ugw.qp_energy[0], atol=1e-4, rtol=1e-4)
        np.testing.assert_allclose(rgw.qp_energy, ugw.qp_energy[1], atol=1e-4, rtol=1e-4)


class Test_scUGW(unittest.TestCase):
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

        mf = mf.density_fit()
        mf.with_df.build()

        cls.mol, cls.mf = mol, mf

    @classmethod
    def tearDownClass(cls):
        del cls.mol, cls.mf

    def test_drpa_regression(self):
        ugw = scUGW(self.mf)
        ugw.polarizability = "drpa"
        ugw.compression = None
        ugw.max_cycle = 250
        ugw.conv_tol_moms = 1e-4
        ugw.conv_tol = 1e-8
        conv, gf, se, _ = ugw.kernel(nmom_max=1)
        self.assertTrue(conv)
        self.assertAlmostEqual(np.max(ugw.qp_energy[0][self.mf.mo_occ[0] > 0]), -0.2816255364, 6)
        self.assertAlmostEqual(np.max(ugw.qp_energy[1][self.mf.mo_occ[1] > 0]), -0.4752488345, 6)
        self.assertAlmostEqual(np.min(ugw.qp_energy[0][self.mf.mo_occ[0] == 0]), 0.1881409112, 6)
        self.assertAlmostEqual(np.min(ugw.qp_energy[1][self.mf.mo_occ[1] == 0]), 0.1892747093, 6)

    def test_dtda_regression(self):
        ugw = scUGW(self.mf)
        ugw.polarizability = "dtda"
        ugw.compression = None
        ugw.max_cycle = 250
        ugw.conv_tol_moms = 1e-4
        ugw.conv_tol = 1e-8
        conv, gf, se, _ = ugw.kernel(nmom_max=1)
        self.assertTrue(conv)
        self.assertAlmostEqual(np.max(ugw.qp_energy[0][self.mf.mo_occ[0] > 0]), -0.2816416308, 6)
        self.assertAlmostEqual(np.max(ugw.qp_energy[1][self.mf.mo_occ[1] > 0]), -0.4747943419, 6)
        self.assertAlmostEqual(np.min(ugw.qp_energy[0][self.mf.mo_occ[0] == 0]), 0.1881707742, 6)
        self.assertAlmostEqual(np.min(ugw.qp_energy[1][self.mf.mo_occ[1] == 0]), 0.1882554684, 6)

    def test_dtda_g0_regression(self):
        ugw = scUGW(self.mf)
        ugw.polarizability = "dtda"
        ugw.compression = None
        ugw.g0 = True
        ugw.max_cycle = 250
        ugw.conv_tol_moms = 1e-4
        ugw.conv_tol = 1e-8
        conv, gf, se, _ = ugw.kernel(nmom_max=3)
        self.assertTrue(conv)
        self.assertAlmostEqual(np.max(ugw.qp_energy[0][self.mf.mo_occ[0] > 0]), -0.2790076125, 6)
        self.assertAlmostEqual(np.max(ugw.qp_energy[1][self.mf.mo_occ[1] > 0]), -0.4709600824, 6)
        self.assertAlmostEqual(np.min(ugw.qp_energy[0][self.mf.mo_occ[0] == 0]), 0.1880377474, 6)
        self.assertAlmostEqual(np.min(ugw.qp_energy[1][self.mf.mo_occ[1] == 0]), 0.1824443000, 6)

    def test_dtda_w0_regression(self):
        ugw = scUGW(self.mf)
        ugw.polarizability = "dtda"
        ugw.compression = None
        ugw.w0 = True
        ugw.max_cycle = 250
        ugw.conv_tol_moms = 1e-4
        ugw.conv_tol = 1e-8
        conv, gf, se, _ = ugw.kernel(nmom_max=3)
        self.assertTrue(conv)
        self.assertAlmostEqual(np.max(ugw.qp_energy[0][self.mf.mo_occ[0] > 0]), -0.2833231253, 6)
        self.assertAlmostEqual(np.max(ugw.qp_energy[1][self.mf.mo_occ[1] > 0]), -0.4758363256, 6)
        self.assertAlmostEqual(np.min(ugw.qp_energy[0][self.mf.mo_occ[0] == 0]), 0.1920524407, 6)
        self.assertAlmostEqual(np.min(ugw.qp_energy[1][self.mf.mo_occ[1] == 0]), 0.1862850716, 6)


if __name__ == "__main__":
    print("Running tests for scUGW")
    unittest.main()
