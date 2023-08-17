"""
Tests for `thc.py`.
"""

import unittest
from os.path import abspath, dirname, join

import numpy as np
import pytest
from pyscf import lib
from pyscf.agf2 import mpi_helper
from pyscf.pbc import dft, gto
from scipy.linalg import cholesky

from momentGW.gw import GW


class Test_THCTDA(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cell = gto.M()
        cell.a = np.eye(3) * 3
        cell.atom = """He 0 0 0; He 1 1 1"""
        cell.basis = "6-31g"
        cell.verbose = 0
        cell.build()

        kpts = cell.make_kpts([1, 1, 1])

        kmf = dft.RKS(cell, xc="pbe")
        kmf.conv_tol = 1e-11
        kmf = kmf.density_fit()
        kmf.exxdiv = None
        kmf.kernel()
        kmf.mo_coeff = mpi_helper.bcast_dict(kmf.mo_coeff, root=0)
        kmf.mo_energy = mpi_helper.bcast_dict(kmf.mo_energy, root=0)

        cls.cell, cls.kmf = cell, kmf

    @classmethod
    def tearDownClass(cls):
        del cls.cell, cls.kmf

    def test_nelec(self):
        gw = GW(self.kmf)
        gw.thc_opts = dict(file_path=abspath(join(dirname(__file__), "..", "tests/thc.h5")))
        gw.polarizability = "thc-dtda"
        conv, gf, se = gw.kernel(nmom_max=1)
        self.assertAlmostEqual(
            gf.make_rdm1().trace(),
            self.cell.nelectron,
            1,
        )

        gw = GW(self.kmf)
        gw.thc_opts = dict(file_path=abspath(join(dirname(__file__), "..", "tests/thc.h5")))
        gw.polarizability = "thc-dtda"
        gw.diagonal_se = True
        gw.vhf_df = False
        conv, gf, se = gw.kernel(nmom_max=1)
        self.assertAlmostEqual(
            gf.make_rdm1().trace(),
            self.cell.nelectron,
            1,
        )

        gw = GW(self.kmf)
        gw.thc_opts = dict(file_path=abspath(join(dirname(__file__), "..", "tests/thc.h5")))
        gw.polarizability = "thc-dtda"
        gw.optimise_chempot = True
        gw.vhf_df = False
        conv, gf, se = gw.kernel(nmom_max=1)
        self.assertAlmostEqual(
            gf.make_rdm1().trace(),
            self.cell.nelectron,
            8,
        )

    def test_moments(self):
        gw = GW(self.kmf)
        gw.thc_opts = dict(file_path=abspath(join(dirname(__file__), "..", "tests/thc.h5")))
        gw.polarizability = "thc-dtda"
        th1, tp1 = gw.build_se_moments(5, gw.ao2mo())
        conv, gf, se = gw.kernel(nmom_max=5)
        th2 = se.get_occupied().moment(range(5))
        tp2 = se.get_virtual().moment(range(5))

        for a, b in zip(th1, th2):
            dif = np.max(np.abs(a - b)) / np.max(np.abs(a))
            self.assertAlmostEqual(dif, 0, 8)
        for a, b in zip(tp1, tp2):
            dif = np.max(np.abs(a - b)) / np.max(np.abs(a))
            self.assertAlmostEqual(dif, 0, 8)

    def test_moments_vs_cderi(self):
        if mpi_helper.size > 1:
            pytest.skip("Doesn't work with MPI")

        gw = GW(self.kmf)
        gw.polarizability = "thc-dtda"
        gw.thc_opts = dict(file_path=abspath(join(dirname(__file__), "..", "tests/thc.h5")))
        th1, tp1 = gw.build_se_moments(5, gw.ao2mo())

        thc_ints = gw.ao2mo()
        thc_ints.transform()
        decou = cholesky(thc_ints.cou, lower=True)

        gw.polarizability = "dtda"
        cd_ints = gw.ao2mo()
        cd_ints._rot = np.eye(decou.shape[0])
        Lpx = lib.einsum("Mp,Mx,ML ->Lpx", thc_ints.Lp, thc_ints.Lp, decou)
        Lia = lib.einsum("Mi,Ma,ML ->Lia", thc_ints.Li, thc_ints.La, decou)
        Lia = Lia.reshape(Lia.shape[0], Lia.shape[1] + Lia.shape[2])
        cd_ints._blocks["Lpx"] = Lpx
        cd_ints._blocks["Lia"] = Lia

        th2, tp2 = gw.build_se_moments(5, cd_ints)

        for a, b in zip(th1, th2):
            dif = np.max(np.abs(a - b)) / np.max(np.abs(a))
            self.assertAlmostEqual(dif, 0, 8)
        for a, b in zip(tp1, tp2):
            dif = np.max(np.abs(a - b)) / np.max(np.abs(a))
            self.assertAlmostEqual(dif, 0, 8)

    def _test_regression(self, xc, kwargs, nmom_max, ip, ea, name=""):
        cell = gto.M()
        cell.a = np.eye(3) * 3
        cell.atom = """He 0 0 0; He 1 1 1"""
        cell.basis = "6-31g"
        cell.verbose = 0
        cell.make_kpts([1, 1, 1])
        cell.build()

        kmf = dft.RKS(cell, xc=xc)
        kmf.conv_tol = 1e-11
        kmf = kmf.density_fit()
        kmf.exxdiv = None
        kmf.kernel()
        kmf.mo_coeff = mpi_helper.bcast_dict(kmf.mo_coeff, root=0)
        kmf.mo_energy = mpi_helper.bcast_dict(kmf.mo_energy, root=0)

        gw = GW(kmf)
        gw.max_cycle = 250
        gw.conv_tol_moms = 1e-4
        gw.conv_tol = 1e-8
        gw.thc_opts = dict(file_path=abspath(join(dirname(__file__), "..", "tests/thc.h5")))
        gw.polarizability = "thc-dtda"
        gw.kernel(nmom_max)
        gw.gf.remove_uncoupled(tol=0.1)
        self.assertTrue(gw.converged)
        self.assertAlmostEqual(gw.gf.get_occupied().energy[-1], ip, 7, msg=name)
        self.assertAlmostEqual(gw.gf.get_virtual().energy[0], ea, 7, msg=name)

    def test_regression_pbe_fock_loop(self):
        ip = -0.2786188906832294
        ea = 1.0822831284078982
        self._test_regression("pbe", dict(), 1, ip, ea, "pbe")


if __name__ == "__main__":
    print("Running tests for THC TDAGW")
    unittest.main()
