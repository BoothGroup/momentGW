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
        mf.mo_coeff = mpi_helper.bcast_dict(mf.mo_coeff, root=0)
        mf.mo_energy = mpi_helper.bcast_dict(mf.mo_energy, root=0)

        mf = mf.density_fit(auxbasis="cc-pv5z-ri")
        mf.with_df.build()

        cls.mol, cls.mf = mol, mf

    @classmethod
    def tearDownClass(cls):
        del cls.mol, cls.mf

    def _test_regression(self, xc, kwargs, nmom_max, ip, ea, ip_full, ea_full, name=""):
        mol = gto.M(atom="H 0 0 0; Li 0 0 1.64", basis="6-31g", verbose=0)
        mf = dft.RKS(mol, xc=xc).density_fit().run()
        mf.mo_coeff = mpi_helper.bcast_dict(mf.mo_coeff, root=0)
        mf.mo_energy = mpi_helper.bcast_dict(mf.mo_energy, root=0)
        gw = qsGW(mf, **kwargs)
        gw.max_cycle = 200
        gw.kernel(nmom_max)
        gw.gf.remove_uncoupled(tol=0.1)
        qp_energy = gw.qp_energy
        self.assertTrue(gw.converged)
        self.assertAlmostEqual(gw.gf.get_occupied().energy[-1], ip_full, 7, msg=name)
        self.assertAlmostEqual(gw.gf.get_virtual().energy[0], ea_full, 7, msg=name)
        self.assertAlmostEqual(np.max(qp_energy[mf.mo_occ > 0]), ip, 7, msg=name)
        self.assertAlmostEqual(np.min(qp_energy[mf.mo_occ == 0]), ea, 7, msg=name)

    def test_regression_simple(self):
        # Quasiparticle energies:
        ip = -0.283719805037
        ea = 0.007318176449
        # GF poles:
        ip_full = -0.265178368463
        ea_full = 0.004998463727
        self._test_regression("hf", dict(), 1, ip, ea, ip_full, ea_full, "simple")

    def test_regression_pbe_srg(self):
        # Quasiparticle energies:
        ip = -0.298283765946
        ea = 0.008369048047
        # GF poles:
        ip_full = -0.418233032000
        ea_full = 0.059983899102
        self._test_regression("pbe", dict(srg=1e-3), 1, ip, ea, ip_full, ea_full, "pbe srg")


if __name__ == "__main__":
    print("Running tests for qsGW")
    unittest.main()
