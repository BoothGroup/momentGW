"""
Tests for `pbc/thc.py`
"""

import unittest
from os.path import abspath, dirname, join

import numpy as np
import pytest
from pyscf import lib
from pyscf.agf2 import mpi_helper
from pyscf.pbc import df, scf, gto
from scipy.linalg import cholesky

from momentGW import KGW

class Test_KGW(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cell = gto.Cell()
        cell.atom = "He 0 0 0; He 1 1 1"
        cell.basis = "6-31g"
        cell.a = np.eye(3) * 3
        cell.verbose = 3
        cell.build()

        kmesh = [2, 2, 2]
        kpts = cell.make_kpts(kmesh)

        with_df = df.DF(cell, kpts=kpts)
        with_df.build()
        kmf = scf.KRKS(cell, xc="pbe", kpts=kpts)
        kmf = kmf.density_fit(auxbasis="weigend")
        kmf.with_df = with_df
        kmf.kernel()

        for k in range(len(kpts)):
            kmf.mo_coeff[k] = mpi_helper.bcast_dict(kmf.mo_coeff[k], root=0)
            kmf.mo_energy[k] = mpi_helper.bcast_dict(kmf.mo_energy[k], root=0)

        cls.cell, cls.kpts, cls.kmf = cell, kpts, kmf

    @classmethod
    def tearDownClass(cls):
        del cls.cell, cls.kpts, cls.kmf

    def test_nelec(self):
        gw = KGW(self.kmf)
        gw.thc_opts = dict(file_path=abspath(join(dirname(__file__), "..", "tests/thc_pbc.h5")))
        gw.polarizability = "thc-dtda"
        conv, gf, se, _ = gw.kernel(nmom_max=1)
        for k in range(gw.nkpts):
            self.assertAlmostEqual(
                gf[k].occupied().moment(0).trace() * 2,
                self.cell.nelectron,
                1,
            )

        gw = KGW(self.kmf)
        gw.thc_opts = dict(file_path=abspath(join(dirname(__file__), "..", "tests/thc_pbc.h5")))
        gw.polarizability = "thc-dtda"
        gw.diagonal_se = True
        gw.vhf_df = False
        conv, gf, se, _ = gw.kernel(nmom_max=1)
        for k in range(gw.nkpts):
            self.assertAlmostEqual(
                gf[k].occupied().moment(0).trace() * 2,
                self.cell.nelectron,
                1,
            )

        gw = KGW(self.kmf)
        gw.thc_opts = dict(file_path=abspath(join(dirname(__file__), "..", "tests/thc_pbc.h5")))
        gw.polarizability = "thc-dtda"
        gw.optimise_chempot = True
        gw.vhf_df = False
        conv, gf, se, _ = gw.kernel(nmom_max=1)
        for k in range(gw.nkpts):
            self.assertAlmostEqual(
                gf[k].occupied().moment(0).trace() * 2,
                self.cell.nelectron,
                1,
            )

    def test_moments(self):
        gw = KGW(self.kmf)
        gw.thc_opts = dict(file_path=abspath(join(dirname(__file__), "..", "tests/thc_pbc.h5")))
        gw.polarizability = "thc-dtda"
        th1, tp1 = gw.build_se_moments(5, gw.ao2mo())
        conv, gf, se, _ = gw.kernel(nmom_max=5)
        for k in range(gw.nkpts):
            th2 = se[k].occupied().moment(range(5))
            tp2 = se[k].virtual().moment(range(5))
            for a, b in zip(th1[k], th2):
                dif = np.max(np.abs(a - b)) / np.max(np.abs(a))
                self.assertAlmostEqual(dif, 0, 8)
            for a, b in zip(tp1[k], tp2):
                dif = np.max(np.abs(a - b)) / np.max(np.abs(a))
                self.assertAlmostEqual(dif, 0, 8)

    def test_moments_vs_cderi(self):
        if mpi_helper.size > 1:
            pytest.skip("Doesn't work with MPI")

        kgw = KGW(self.kmf)
        kgw.polarizability = "thc-dtda"
        kgw.thc_opts = dict(file_path=abspath(join(dirname(__file__), "..", "tests/thc_pbc.h5")))
        th1, tp1 = kgw.build_se_moments(5, kgw.ao2mo())

        thc_ints = kgw.ao2mo()
        kpts = thc_ints.kpts
        thc_ints.transform()

        decou = {}
        Lpx = {}
        Lia = {}
        for q in thc_ints.kpts.loop(1):
            decou[q] = cholesky(thc_ints.cou[q], lower=True)

        kgw.polarizability = "dtda"
        cd_ints = kgw.ao2mo()
        cd_ints._rot = np.eye(decou[0].shape[0])
        for q in kgw.kpts.loop(1):
            for kj in kpts.loop(1, mpi=True):
                kb = kpts.member(kpts.wrap_around(kpts[q] + kpts[kj]))
                Lpx[kj, kb] = lib.einsum("Mp,Mx,ML ->Lpx", thc_ints.Lp[kj], thc_ints.Lp[kb], decou[q])
                temp = lib.einsum("Mi,Ma,ML ->Lia", thc_ints.Li[kj], thc_ints.La[kb], decou[q])
                Lia[kj, kb] = temp.reshape(temp.shape[0], temp.shape[1] + temp.shape[2])

        cd_ints._blocks["Lpx"] = Lpx
        cd_ints._blocks["Lia"] = Lia
        cd_ints._blocks["Lai"] = Lia

        th2, tp2 = kgw.build_se_moments(5, cd_ints)

        for a, b in zip(th1, th2):
            dif = np.max(np.abs(a - b)) / np.max(np.abs(a))
            self.assertAlmostEqual(dif, 0, 8)
        for a, b in zip(tp1, tp2):
            dif = np.max(np.abs(a - b)) / np.max(np.abs(a))
            self.assertAlmostEqual(dif, 0, 8)

    def _test_regression(self, xc, kwargs, nmom_max, ip, ea, name=""):
        cell = gto.Cell()
        cell.atom = "He 0 0 0; He 1 1 1"
        cell.basis = "6-31g"
        cell.a = np.eye(3) * 3
        cell.verbose = 3
        cell.build()

        kmesh = [2, 2, 2]
        kpts = cell.make_kpts(kmesh)

        with_df = df.DF(cell, kpts=kpts)
        with_df.build()
        kmf = scf.KRKS(cell, xc="pbe", kpts=kpts)
        kmf = kmf.density_fit(auxbasis="weigend")
        kmf.with_df = with_df
        kmf.kernel()

        for k in range(len(kpts)):
            kmf.mo_coeff[k] = mpi_helper.bcast_dict(kmf.mo_coeff[k], root=0)
            kmf.mo_energy[k] = mpi_helper.bcast_dict(kmf.mo_energy[k], root=0)

        kgw = KGW(kmf)
        kgw.max_cycle = 250
        kgw.conv_tol_moms = 1e-4
        kgw.conv_tol = 1e-8
        kgw.thc_opts = dict(file_path=abspath(join(dirname(__file__), "..", "tests/thc_pbc.h5")))
        kgw.polarizability = "thc-dtda"
        kgw.kernel(nmom_max)
        for k in range(len(kpts)):
            gf = kgw.gf[k].physical(weight=0.1)
            self.assertTrue(kgw.converged)
            self.assertAlmostEqual(gf.occupied().energies[-1], ip[k], 7, msg=name)
            self.assertAlmostEqual(gf.virtual().energies[0], ea[k], 7, msg=name)

    def test_regression_pbe_fock_loop(self):
        ip = [-0.48261234253482393, -0.5020081305060984, -0.5020430194398028, -0.5098036236597017, -0.5020214631928213,
              -0.5097884459452672, -0.5098096687525852, -0.5116661568677767]
        ea = [1.0832748083365689, 1.278552330973248, 1.2785590341544015, 1.451159741552717, 1.2785545610761493,
              1.4511524920016912, 1.4511646728150698, 1.5094991289412134]

        self._test_regression("pbe", dict(), 1, ip, ea, "pbe")

if __name__ == "__main__":
    print("Running tests for THC TDAGW")
    unittest.main()
