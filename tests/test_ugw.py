"""
Tests for `uhf/gw.py`.
"""

import unittest

import numpy as np
import pytest
from pyscf import dft, gto, gw, lib, tdscf
from pyscf.agf2 import mpi_helper

from momentGW import GW, UGW


class Test_UGW_vs_RGW(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        mol = gto.Mole()
        mol.atom = "Li 0 0 0; H 0 0 1.64"
        mol.basis = "cc-pvdz"
        mol.verbose = 0
        mol.build()

        mf = dft.RKS(mol)
        mf.xc = "hf"
        mf.conv_tol = 1e-12
        mf = mf.density_fit(auxbasis="cc-pvdz-ri")
        mf.with_df.build()
        mf.kernel()

        mf.mo_coeff = mpi_helper.bcast_dict(mf.mo_coeff, root=0)
        mf.mo_energy = mpi_helper.bcast_dict(mf.mo_energy, root=0)

        cls.mol, cls.mf = mol, mf

    @classmethod
    def tearDownClass(cls):
        del cls.mol, cls.mf

    def test_dtda(self):
        rgw = GW(self.mf)
        rgw.compression = None
        rgw.polarizability = "dtda"
        rgw.kernel(5)

        uhf = self.mf.to_uks()
        uhf.with_df = self.mf.with_df

        ugw = UGW(uhf)
        ugw.compression = None
        ugw.polarizability = "dtda"
        ugw.kernel(5)

        self.assertTrue(rgw.converged)
        self.assertTrue(ugw.converged)

        np.testing.assert_allclose(rgw.qp_energy, ugw.qp_energy[0])
        np.testing.assert_allclose(rgw.qp_energy, ugw.qp_energy[1])

    def test_dtda_fock_loop(self):
        rgw = GW(self.mf)
        rgw.compression = None
        rgw.polarizability = "dtda"
        rgw.fock_loop = True
        rgw.conv_tol_nelec = 1e-10
        rgw.conv_tol_rdm1 = 1e-12
        rgw.kernel(5)

        uhf = self.mf.to_uks()
        uhf.with_df = self.mf.with_df

        ugw = UGW(uhf)
        ugw.compression = None
        ugw.polarizability = "dtda"
        ugw.fock_loop = True
        ugw.conv_tol_nelec = 1e-10
        ugw.conv_tol_rdm1 = 1e-12
        ugw.kernel(5)

        self.assertTrue(rgw.converged)
        self.assertTrue(ugw.converged)

        np.testing.assert_allclose(rgw.qp_energy, ugw.qp_energy[0], atol=1e-8, rtol=1e-6)
        np.testing.assert_allclose(rgw.qp_energy, ugw.qp_energy[1], atol=1e-8, rtol=1e-6)

    def test_drpa(self):
        rgw = GW(self.mf)
        rgw.compression = None
        rgw.polarizability = "drpa"
        rgw.kernel(5)

        uhf = self.mf.to_uks()
        uhf.with_df = self.mf.with_df

        ugw = UGW(uhf)
        ugw.compression = None
        ugw.polarizability = "drpa"
        ugw.kernel(5)

        self.assertTrue(rgw.converged)
        self.assertTrue(ugw.converged)

        np.testing.assert_allclose(rgw.qp_energy, ugw.qp_energy[0])
        np.testing.assert_allclose(rgw.qp_energy, ugw.qp_energy[1])

    def test_dtda_compression(self):
        rgw = GW(self.mf)
        rgw.compression = "ov,oo"
        rgw.compression_tol = 1e-4
        rgw.polarizability = "dtda"
        rgw.kernel(5)

        uhf = self.mf.to_uks()
        uhf.with_df = self.mf.with_df

        ugw = UGW(uhf)
        ugw.compression = "ov,oo"
        ugw.compression_tol = 1e-4
        ugw.polarizability = "dtda"
        ugw.kernel(5)

        self.assertTrue(rgw.converged)
        self.assertTrue(ugw.converged)

        np.testing.assert_allclose(rgw.qp_energy, ugw.qp_energy[0])
        np.testing.assert_allclose(rgw.qp_energy, ugw.qp_energy[1])

    def test_drpa_compression(self):
        rgw = GW(self.mf)
        rgw.compression = "ia,oo"
        rgw.compression_tol = 1e-4
        rgw.polarizability = "drpa"
        rgw.kernel(5)

        uhf = self.mf.to_uks()
        uhf.with_df = self.mf.with_df

        ugw = UGW(uhf)
        ugw.compression = "ia,oo"
        ugw.compression_tol = 1e-4
        ugw.polarizability = "drpa"
        ugw.kernel(5)

        self.assertTrue(rgw.converged)
        self.assertTrue(ugw.converged)

        np.testing.assert_allclose(rgw.qp_energy, ugw.qp_energy[0])
        np.testing.assert_allclose(rgw.qp_energy, ugw.qp_energy[1])

    def test_dtda_frozen(self):
        rgw = GW(self.mf)
        rgw.polarizability = "dtda"
        rgw.frozen = [0]
        rgw.kernel(5)

        uhf = self.mf.to_uks()
        uhf.with_df = self.mf.with_df

        ugw = UGW(uhf)
        ugw.polarizability = "dtda"
        ugw.frozen = [0]
        ugw.kernel(5)

        self.assertTrue(rgw.converged)
        self.assertTrue(ugw.converged)

        np.testing.assert_allclose(rgw.qp_energy, ugw.qp_energy[0])
        np.testing.assert_allclose(rgw.qp_energy, ugw.qp_energy[1])

    def test_drpa_frozen(self):
        rgw = GW(self.mf)
        rgw.frozen = [0]
        rgw.kernel(5)

        uhf = self.mf.to_uks()
        uhf.with_df = self.mf.with_df

        ugw = UGW(uhf)
        ugw.frozen = [0]
        ugw.kernel(5)

        self.assertTrue(rgw.converged)
        self.assertTrue(ugw.converged)

        np.testing.assert_allclose(rgw.qp_energy, ugw.qp_energy[0])
        np.testing.assert_allclose(rgw.qp_energy, ugw.qp_energy[1])


class Test_UGW(unittest.TestCase):
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

    # TODO
    # def test_vs_pyscf_dtda(self):
    #    pass

    def test_vs_pyscf_drpa(self):
        ugw = UGW(self.mf)
        ugw.compression = None
        ugw.npoints = 128
        conv, gf, se, _ = ugw.kernel(nmom_max=9)
        ugw_exact = gw.ugw_ac.UGWAC(self.mf)
        ugw_exact.kernel()
        self.assertAlmostEqual(
            ugw.qp_energy[0][ugw_exact.mo_occ[0] > 0].max(),
            ugw_exact.mo_energy[0][ugw_exact.mo_occ[0] > 0].max(),
            2,
        )
        self.assertAlmostEqual(
            ugw.qp_energy[0][ugw_exact.mo_occ[0] == 0].min(),
            ugw_exact.mo_energy[0][ugw_exact.mo_occ[0] == 0].min(),
            2,
        )
        self.assertAlmostEqual(
            ugw.qp_energy[1][ugw_exact.mo_occ[1] > 0].max(),
            ugw_exact.mo_energy[1][ugw_exact.mo_occ[1] > 0].max(),
            2,
        )
        self.assertAlmostEqual(
            ugw.qp_energy[1][ugw_exact.mo_occ[1] == 0].min(),
            ugw_exact.mo_energy[1][ugw_exact.mo_occ[1] == 0].min(),
            2,
        )


class Test_UGW_no_beta(unittest.TestCase):
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
        ugw = UGW(self.mf)
        ugw.polarizability = "dtda"
        ugw.compression = None
        ugw.npoints = 128
        conv, gf, se, _ = ugw.kernel(nmom_max=9)
        self.assertAlmostEqual(lib.fp(ugw.qp_energy[0]), -1.2080078778)
        self.assertAlmostEqual(lib.fp(ugw.qp_energy[1]), -0.5204708092)

    def test_drpa_regression(self):
        ugw = UGW(self.mf)
        ugw.compression = None
        ugw.npoints = 128
        conv, gf, se, _ = ugw.kernel(nmom_max=9)
        self.assertAlmostEqual(lib.fp(ugw.qp_energy[0]), -1.2014130154)
        self.assertAlmostEqual(lib.fp(ugw.qp_energy[1]), -0.5116967415)


if __name__ == "__main__":
    print("Running tests for UGW")
    unittest.main()
