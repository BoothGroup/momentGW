"""
Tests for `qsgw.py`.
"""

import unittest

import numpy as np
import pytest
from pyscf import dft, gto
from pyscf.agf2 import mpi_helper

from momentGW import qsGW


class Test_qsGW(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        mol = gto.Mole()
        mol.atom = "Li 0 0 0; H 0 0 1.64"
        mol.basis = "cc-pvdz"
        mol.verbose = 0
        mol.build()

        mf = dft.RKS(mol)
        mf.xc = "hf"
        mf.conv_tol = 1e-11
        mf.kernel()

        mf = mf.density_fit(auxbasis="cc-pv5z-ri")
        mf.with_df.build()

        cls.mol, cls.mf = mol, mf

    @classmethod
    def tearDownClass(cls):
        del cls.mol, cls.mf

    def test_nelec(self):
        gw = qsGW(self.mf)
        gw.diagonal_se = True
        gw.vhf_df = False
        conv, gf, se = gw.kernel(nmom_max=1)
        self.assertAlmostEqual(
            gf.make_rdm1().trace(),
            self.mol.nelectron,
            1,
        )
        gw.optimise_chempot = True
        gw.vhf_df = False
        conv, gf, se = gw.kernel(nmom_max=1)
        self.assertAlmostEqual(
            gf.make_rdm1().trace(),
            self.mol.nelectron,
            8,
        )

    def _test_regression(self, xc, kwargs, nmom_max, ip, ea, name=""):
        mol = gto.M(atom="H 0 0 0; Li 0 0 1.64", basis="6-31g", verbose=0)
        mf = dft.RKS(mol, xc=xc).density_fit().run()
        gw = qsGW(mf, **kwargs)
        gw.kernel(nmom_max)
        gw.gf.remove_uncoupled(tol=0.1)
        self.assertAlmostEqual(gw.gf.get_occupied().energy[-1], ip, 7, msg=name)
        self.assertAlmostEqual(gw.gf.get_virtual().energy[0], ea, 7, msg=name)

    def test_regression_simple(self):
        ip = -0.283719805037
        ea = 0.007318176449
        self._test_regression("hf", dict(), 1, ip, ea, "simple")

    def test_regression_pbe_srg(self):
        ip = -0.298283765946
        ea = 0.008369048047
        self._test_regression("pbe", dict(srg=1e-3), 1, ip, ea, "pbe srg")


if __name__ == "__main__":
    print("Running tests for qsGW")
    unittest.main()
