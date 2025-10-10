"""Tests for `pbc/uhf/qsgw.py`"""

import unittest

import numpy as np
from pyscf.agf2 import mpi_helper
from pyscf.pbc import dft, gto
from pyscf.pbc.tools import k2gamma

from momentGW import qsKGW, qsKUGW


class Test_qsKUGW_vs_qsKGW(unittest.TestCase):
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
        krgw = qsKGW(self.mf)
        krgw.compression = None
        krgw.polarizability = "dtda"
        krgw.kernel(3)

        uhf = self.mf.to_uhf()
        uhf.with_df = self.mf.with_df

        kugw = qsKUGW(uhf)
        kugw.compression = None
        kugw.polarizability = "dtda"
        kugw.kernel(3)

        self.assertTrue(krgw.converged)
        self.assertTrue(kugw.converged)

        np.testing.assert_allclose(krgw.qp_energy, kugw.qp_energy[0])
        np.testing.assert_allclose(krgw.qp_energy, kugw.qp_energy[1])


# FIXME bad convergence...
# class Test_qsKUGW_no_beta(unittest.TestCase):
#    @classmethod
#    def setUpClass(cls):
#        cell = gto.Cell()
#        cell.atom = "H 0 0 0; H 1 1 1"
#        cell.basis = "sto3g"
#        cell.charge = 1
#        cell.spin = 1
#        cell.a = np.eye(3) * 3
#        cell.max_memory = 1e10
#        cell.verbose = 0
#        cell.build()
#
#        kmesh = [3, 1, 1]
#        kpts = cell.make_kpts(kmesh)
#
#        mf = dft.KUKS(cell, kpts, xc="hf")
#        mf = mf.density_fit(auxbasis="weigend")
#        mf.with_df.force_dm_kbuild = True
#        mf.exxdiv = None
#        mf.conv_tol = 1e-10
#        mf.kernel()
#
#        for s in range(2):
#            for k in range(len(kpts)):
#                mf.mo_coeff[s][k] = mpi_helper.bcast_dict(mf.mo_coeff[s][k], root=0)
#                mf.mo_energy[s][k] = mpi_helper.bcast_dict(mf.mo_energy[s][k], root=0)
#
#        cls.cell, cls.kpts, cls.mf = cell, kpts, mf
#
#    @classmethod
#    def tearDownClass(cls):
#        del cls.cell, cls.kpts, cls.mf
#
#    def test_dtda_regression(self):
#        kugw = qsKUGW(self.mf, verbose=4)
#        kugw.compression = None
#        kugw.polarizability = "dtda"
#        kugw.kernel(1)
#
#        self.assertTrue(kugw.converged)
#
#        self.assertAlmostEqual(lib.fp(kugw.qp_energy[0]), -0.0042127651)
#        self.assertAlmostEqual(lib.fp(kugw.qp_energy[1]), -0.0785013870)


if __name__ == "__main__":
    print("Running tests for qsKUGW")
    unittest.main()
