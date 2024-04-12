"""
Tests for `uhf/qsgw.py`.
"""

import unittest

import numpy as np
import pytest
from pyscf import dft, gto, gw, lib, tdscf
from pyscf.agf2 import mpi_helper

from momentGW import qsGW
from momentGW.uhf import qsUGW


class Test_qsUGW_vs_qsRGW(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        mol = gto.Mole()
        mol.atom = "Li 0 0 0; H 0 0 1.64"
        mol.basis = "6-31g"
        mol.verbose = 0
        mol.build()

        mf = dft.RKS(mol)
        mf.xc = "hf"
        mf.conv_tol = 1e-10
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
        rgw = qsGW(self.mf)
        rgw.compression = None
        rgw.polarizability = "dtda"
        rgw.kernel(3)

        uhf = self.mf.to_uks()
        uhf.with_df = self.mf.with_df

        ugw = qsUGW(uhf)
        ugw.compression = None
        ugw.polarizability = "dtda"
        ugw.kernel(3)

        self.assertTrue(rgw.converged)
        self.assertTrue(ugw.converged)

        np.testing.assert_allclose(rgw.qp_energy, ugw.qp_energy[0])
        np.testing.assert_allclose(rgw.qp_energy, ugw.qp_energy[1])

    def test_drpa_compression(self):
        rgw = qsGW(self.mf)
        rgw.compression = "ia,oo"
        rgw.compression_tol = 1e-4
        rgw.polarizability = "drpa"
        rgw.kernel(1)

        uhf = self.mf.to_uks()
        uhf.with_df = self.mf.with_df

        ugw = qsUGW(uhf)
        ugw.compression = "ia,oo"
        ugw.compression_tol = 1e-4
        ugw.polarizability = "drpa"
        ugw.kernel(1)

        self.assertTrue(rgw.converged)
        self.assertTrue(ugw.converged)

        np.testing.assert_allclose(rgw.qp_energy, ugw.qp_energy[0], atol=1e-8, rtol=1e-6)
        np.testing.assert_allclose(rgw.qp_energy, ugw.qp_energy[1], atol=1e-8, rtol=1e-6)


class Test_qsUGW(unittest.TestCase):
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
        ugw = qsUGW(self.mf)
        ugw.polarizability = "drpa"
        ugw.compression = None
        ugw.npoints = 128
        conv, gf, se, _ = ugw.kernel(nmom_max=3)
        self.assertTrue(conv)
        self.assertAlmostEqual(np.max(ugw.qp_energy[0][self.mf.mo_occ[0] > 0]), -0.2818994392, 6)
        self.assertAlmostEqual(np.max(ugw.qp_energy[1][self.mf.mo_occ[1] > 0]), -0.4757294661, 6)
        self.assertAlmostEqual(np.min(ugw.qp_energy[0][self.mf.mo_occ[0] == 0]), 0.1916730746, 6)
        self.assertAlmostEqual(np.min(ugw.qp_energy[1][self.mf.mo_occ[1] == 0]), 0.1841548976, 6)

    def test_dtda_regression(self):
        ugw = qsUGW(self.mf)
        ugw.polarizability = "dtda"
        ugw.compression = None
        conv, gf, se, _ = ugw.kernel(nmom_max=3)
        self.assertTrue(conv)
        self.assertAlmostEqual(np.max(ugw.qp_energy[0][self.mf.mo_occ[0] > 0]), -0.2815868529, 6)
        self.assertAlmostEqual(np.max(ugw.qp_energy[1][self.mf.mo_occ[1] > 0]), -0.4760636064, 6)
        self.assertAlmostEqual(np.min(ugw.qp_energy[0][self.mf.mo_occ[0] == 0]), 0.1921630081, 6)
        self.assertAlmostEqual(np.min(ugw.qp_energy[1][self.mf.mo_occ[1] == 0]), 0.1840942070, 6)


class Test_qsUGW_no_beta(unittest.TestCase):
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
        ugw = qsUGW(self.mf)
        ugw.polarizability = "dtda"
        ugw.compression = None
        ugw.npoints = 128
        conv, gf, se, _ = ugw.kernel(nmom_max=9)
        self.assertAlmostEqual(lib.fp(ugw.qp_energy[0]), -1.2077826593)
        self.assertAlmostEqual(lib.fp(ugw.qp_energy[1]), -0.5195654842)

    def test_drpa_regression(self):
        ugw = qsUGW(self.mf)
        ugw.compression = None
        ugw.npoints = 128
        conv, gf, se, _ = ugw.kernel(nmom_max=9)
        self.assertAlmostEqual(lib.fp(ugw.qp_energy[0]), -1.2028363871)
        self.assertAlmostEqual(lib.fp(ugw.qp_energy[1]), -0.5121796398)


if __name__ == "__main__":
    print("Running tests for qsUGW")
    unittest.main()
