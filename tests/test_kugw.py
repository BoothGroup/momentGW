"""
Tests for `pbc/uhf/gw.py`
"""

import unittest

import numpy as np
from pyscf import lib
from pyscf.agf2 import mpi_helper
from pyscf.pbc import dft, gto
from pyscf.pbc.tools import k2gamma

from momentGW import GW, KGW, KUGW, UGW


class Test_KUGW_vs_KRGW(unittest.TestCase):
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
        smf.with_df.force_dm_kbuild = True

        cls.cell, cls.kpts, cls.mf, cls.smf = cell, kpts, mf, smf

    @classmethod
    def tearDownClass(cls):
        del cls.cell, cls.kpts, cls.mf, cls.smf

    def test_dtda(self):
        krgw = KGW(self.mf)
        krgw.compression = None
        krgw.polarizability = "dtda"
        krgw.kernel(3)

        uhf = self.mf.to_uhf()
        uhf.with_df = self.mf.with_df

        kugw = KUGW(uhf)
        kugw.compression = None
        kugw.polarizability = "dtda"
        kugw.kernel(3)

        self.assertTrue(krgw.converged)
        self.assertTrue(kugw.converged)

        np.testing.assert_allclose(krgw.qp_energy, kugw.qp_energy[0])
        np.testing.assert_allclose(krgw.qp_energy, kugw.qp_energy[1])

    def test_dtda_fock_loop(self):
        krgw = KGW(self.mf)
        krgw.compression = None
        krgw.polarizability = "dtda"
        krgw.fock_loop = True
        krgw.conv_tol_nelec = 1e-8
        krgw.conv_tol_rdm1 = 1e-10
        krgw.kernel(3)

        uhf = self.mf.to_uhf()
        uhf.with_df = self.mf.with_df

        kugw = KUGW(uhf)
        kugw.compression = None
        kugw.polarizability = "dtda"
        kugw.fock_loop = True
        kugw.conv_tol_nelec = 1e-8
        kugw.conv_tol_rdm1 = 1e-10
        kugw.kernel(3)

        self.assertTrue(krgw.converged)
        self.assertTrue(kugw.converged)

        np.testing.assert_allclose(krgw.qp_energy, kugw.qp_energy[0], atol=1e-8, rtol=1e-6)
        np.testing.assert_allclose(krgw.qp_energy, kugw.qp_energy[1], atol=1e-8, rtol=1e-6)

    # def test_dtda_compression(self):
    #    krgw = KGW(self.mf)
    #    krgw.compression = "ov,oo"
    #    krgw.compression_tol = 1e-4
    #    krgw.polarizability = "dtda"
    #    krgw.kernel(3)

    #    uhf = self.mf.to_uhf()
    #    uhf.with_df = self.mf.with_df

    #    kugw = KUGW(uhf)
    #    kugw.compression = "ov,oo"
    #    kugw.compression_tol = 1e-4
    #    kugw.polarizability = "dtda"
    #    kugw.kernel(3)

    #    self.assertTrue(krgw.converged)
    #    self.assertTrue(kugw.converged)

    #    np.testing.assert_allclose(krgw.qp_energy, kugw.qp_energy[0])
    #    np.testing.assert_allclose(krgw.qp_energy, kugw.qp_energy[1])


class Test_KUGW(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cell = gto.Cell()
        cell.atom = "He 0 0 0; He 1 1 1"
        cell.basis = "6-31g"
        cell.spin = 2
        cell.a = np.eye(3) * 3
        # cell.atom = "H 0 0 0; H 0.75 0 0"
        # cell.basis = "6-31g"
        # cell.spin = 2
        # cell.a = [[1.5, 0, 0], [0, 25, 0], [0, 0, 25]]
        cell.max_memory = 1e10
        cell.verbose = 0
        cell.build()

        kmesh = [3, 1, 1]
        kpts = cell.make_kpts(kmesh, time_reversal_symmetry=True)

        mf = dft.KUKS(cell, kpts, xc="hf")
        mf = mf.density_fit(auxbasis="weigend")
        mf.with_df.force_dm_kbuild = True
        mf.exxdiv = None
        mf.conv_tol = 1e-10
        mf.kernel()

        mf.mo_energy = kpts.transform_mo_energy(mf.mo_energy)
        mf.mo_coeff = kpts.transform_mo_coeff(mf.mo_coeff)
        mf.mo_occ = kpts.transform_mo_occ(mf.mo_occ)
        mf.kpts = kpts.kpts

        for s in range(2):
            for k in range(len(mf.kpts)):
                mf.mo_coeff[s][k] = mpi_helper.bcast_dict(mf.mo_coeff[s][k], root=0)
                mf.mo_energy[s][k] = mpi_helper.bcast_dict(mf.mo_energy[s][k], root=0)

        smf = k2gamma.k2gamma(mf, kmesh=kmesh)
        smf = smf.density_fit(auxbasis="weigend")
        smf.exxdiv = None
        smf.with_df.force_dm_kbuild = True

        cls.cell, cls.kpts, cls.mf, cls.smf = cell, kpts, mf, smf

    @classmethod
    def tearDownClass(cls):
        del cls.cell, cls.kpts, cls.mf, cls.smf

    def _test_vs_supercell(self, gw, kgw, tol=1e-8):
        e1 = np.sort(gw.qp_energy)
        e2 = (
            np.sort(np.concatenate(kgw.qp_energy[0])),
            np.sort(np.concatenate(kgw.qp_energy[1])),
        )
        np.testing.assert_allclose(e1, e2, atol=tol)

    def test_dtda_vs_supercell(self):
        nmom_max = 3

        kgw = KUGW(self.mf)
        kgw.polarizability = "dtda"
        kgw.kernel(nmom_max)

        gw = UGW(self.smf)
        gw.polarizability = "dtda"
        gw.kernel(nmom_max)

        self._test_vs_supercell(gw, kgw)


class Test_KUGW_no_beta(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cell = gto.Cell()
        cell.atom = "H 0 0 0; H 0.75 0 0"
        cell.basis = "sto3g"
        cell.charge = 0
        cell.spin = 2
        cell.a = [[1.5, 0, 0], [0, 25, 0], [0, 0, 25]]
        cell.max_memory = 1e10
        cell.verbose = 0
        cell.precision = 1e-14
        cell.build()

        kmesh = [3, 1, 1]
        kpts = cell.make_kpts(kmesh)

        mf = dft.KUKS(cell, kpts, xc="hf")
        mf = mf.density_fit(auxbasis="weigend")
        mf.with_df.force_dm_kbuild = True
        mf.exxdiv = None
        mf.conv_tol = 1e-11
        # mf.kernel()

        # Avoid unstable system:
        mf.converged = True
        mf.mo_occ = (
            np.array([[1, 0], [1, 0], [1, 0]]),
            np.array([[1, 0], [1, 1], [0, 0]]),
        )
        mf.mo_energy = (
            np.array([[-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0]]),
            np.array([[-1.0, 1.0], [-1.0, -0.5], [0.5, 1.0]]),
        )
        mf.mo_coeff = (
            np.array([np.eye(2)] * 3),
            np.array([np.eye(2)] * 3),
        )

        cls.cell, cls.kpts, cls.mf = cell, kpts, mf

    @classmethod
    def tearDownClass(cls):
        del cls.cell, cls.kpts, cls.mf

    def test_dtda_regression(self):
        kugw = KUGW(self.mf)
        kugw.compression = None
        kugw.polarizability = "dtda"
        kugw.kernel(1)

        self.assertTrue(kugw.converged)

        self.assertAlmostEqual(lib.fp(kugw.qp_energy[0]), -0.0339794308)
        self.assertAlmostEqual(lib.fp(kugw.qp_energy[1]), 0.3341749639)


if __name__ == "__main__":
    print("Running tests for KUGW")
    unittest.main()
