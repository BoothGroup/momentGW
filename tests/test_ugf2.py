"""
Tests for `uhf/gf2.py`.
"""

import unittest

import numpy as np
import pytest
from pyscf import scf, gto, agf2, lib
from pyscf.agf2 import mpi_helper

from momentGW import GF2, G0F2, evGF2, qsGF2, fsGF2
from momentGW.uhf import UGF2, UG0F2, evUGF2, qsUGF2, fsUGF2


class Test_UGF2_vs_RGF2(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        mol = gto.Mole()
        mol.atom = "Li 0 0 0; H 0 0 1.64"
        mol.basis = "6-31g"
        mol.verbose = 0
        mol.build()

        mf = scf.RHF(mol)
        mf = mf.density_fit()
        mf.kernel()

        mf.mo_coeff = mpi_helper.bcast_dict(mf.mo_coeff, root=0)
        mf.mo_energy = mpi_helper.bcast_dict(mf.mo_energy, root=0)

        cls.mol, cls.mf = mol, mf

    @classmethod
    def tearDownClass(cls):
        del cls.mol, cls.mf

    def test_gf2(self):
        rgf2 = GF2(self.mf)
        rgf2.fock_loop = True
        rgf2.kernel(1)

        uhf = self.mf.to_uhf()
        uhf.with_df = self.mf.with_df

        ugf2 = UGF2(uhf)
        ugf2.fock_loop = True
        ugf2.kernel(1)

        self.assertTrue(rgf2.converged)
        self.assertTrue(ugf2.converged)

        np.testing.assert_allclose(rgf2.qp_energy, ugf2.qp_energy[0])
        np.testing.assert_allclose(rgf2.qp_energy, ugf2.qp_energy[1])

    def test_g0f2(self):
        rgf2 = G0F2(self.mf)
        rgf2.kernel(3)

        uhf = self.mf.to_uhf()
        uhf.with_df = self.mf.with_df

        ugf2 = UG0F2(uhf)
        ugf2.kernel(3)

        self.assertTrue(rgf2.converged)
        self.assertTrue(ugf2.converged)

        np.testing.assert_allclose(rgf2.qp_energy, ugf2.qp_energy[0])
        np.testing.assert_allclose(rgf2.qp_energy, ugf2.qp_energy[1])

    def test_evgf2(self):
        rgf2 = evGF2(self.mf)
        rgf2.kernel(3)

        uhf = self.mf.to_uhf()
        uhf.with_df = self.mf.with_df

        ugf2 = evUGF2(uhf)
        ugf2.kernel(3)

        self.assertTrue(rgf2.converged)
        self.assertTrue(ugf2.converged)

        np.testing.assert_allclose(rgf2.qp_energy, ugf2.qp_energy[0])
        np.testing.assert_allclose(rgf2.qp_energy, ugf2.qp_energy[1])

    def test_qsgf2(self):
        rgf2 = qsGF2(self.mf)
        rgf2.kernel(3)

        uhf = self.mf.to_uhf()
        uhf.with_df = self.mf.with_df

        ugf2 = qsUGF2(uhf)
        ugf2.kernel(3)

        self.assertTrue(rgf2.converged)
        self.assertTrue(ugf2.converged)

        np.testing.assert_allclose(rgf2.qp_energy, ugf2.qp_energy[0])
        np.testing.assert_allclose(rgf2.qp_energy, ugf2.qp_energy[1])

    def test_fsgf2(self):
        rgf2 = fsGF2(self.mf)
        rgf2.kernel(3)

        uhf = self.mf.to_uhf()
        uhf.with_df = self.mf.with_df

        ugf2 = fsUGF2(uhf)
        ugf2.kernel(3)

        self.assertTrue(rgf2.converged)
        self.assertTrue(ugf2.converged)

        np.testing.assert_allclose(rgf2.qp_energy, ugf2.qp_energy[0])
        np.testing.assert_allclose(rgf2.qp_energy, ugf2.qp_energy[1])


class Test_UGF2_no_beta(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        mol = gto.Mole()
        mol.atom = "H 0 0 0; H 0 0 1"
        mol.basis = "sto3g"
        mol.spin = 1
        mol.charge = 1
        mol.verbose = 0
        mol.build()

        mf = scf.UHF(mol)
        mf = mf.density_fit()
        mf.conv_tol = 1e-11
        mf.kernel()

        mf.mo_coeff = (
            mpi_helper.bcast_dict(mf.mo_coeff[0], root=0),
            mpi_helper.bcast_dict(mf.mo_coeff[1], root=0),
        )
        mf.mo_energy = (
            mpi_helper.bcast_dict(mf.mo_energy[0], root=0),
            mpi_helper.bcast_dict(mf.mo_energy[1], root=0),
        )

        cls.mol, cls.mf = mol, mf

    @classmethod
    def tearDownClass(cls):
        del cls.mol, cls.mf

    def test_gf2_regression(self):
        gf2 = UGF2(self.mf)
        gf2.fock_loop = True
        gf2.kernel(1)
        self.assertAlmostEqual(lib.fp(gf2.qp_energy[0]), -1.199526464517)
        self.assertAlmostEqual(lib.fp(gf2.qp_energy[1]), -0.494928342857)

    def test_evgf2_regression(self):
        gf2 = evUGF2(self.mf)
        gf2.kernel(1)
        self.assertAlmostEqual(lib.fp(gf2.qp_energy[0]), -1.429147616621)
        self.assertAlmostEqual(lib.fp(gf2.qp_energy[1]), -1.361677731090)


if __name__ == "__main__":
    print("Running tests for UGF2")
    unittest.main()
