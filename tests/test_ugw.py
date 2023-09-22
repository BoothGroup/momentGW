"""
Tests for `uhf/gw.py`.
"""

import unittest

import numpy as np
import pytest
from pyscf import dft, gto, gw, lib, tdscf
from pyscf.agf2 import mpi_helper

from momentGW import GW
from momentGW.uhf import UGW


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
        rgw = GW(self.mf)
        rgw.polarizability = "dtda"
        rgw.kernel(5)

        uhf = self.mf.to_uks()
        uhf.with_df = self.mf.with_df

        ugw = UGW(uhf)
        ugw.polarizability = "dtda"
        ugw.kernel(5)

        self.assertTrue(rgw.converged)
        self.assertTrue(ugw.converged)

        np.testing.assert_allclose(rgw.qp_energy, ugw.qp_energy[0])
        np.testing.assert_allclose(rgw.qp_energy, ugw.qp_energy[1])

    def test_drpa(self):
        rgw = GW(self.mf)
        rgw.polarizability = "drpa"
        rgw.kernel(5)

        uhf = self.mf.to_uks()
        uhf.with_df = self.mf.with_df

        ugw = UGW(uhf)
        ugw.polarizability = "drpa"
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
    #def test_vs_pyscf_dtda(self):
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


if __name__ == "__main__":
    print("Running tests for UGW")
    unittest.main()
