"""
Tests for `gf2.py`.
"""

import unittest

import numpy as np
import pytest
from pyscf import scf, gto, agf2
from pyscf.agf2 import mpi_helper

from momentGW import GF2


class Test_GF2(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        mol = gto.Mole()
        mol.atom = "H 0 0 0; Li 0 0 1.64"
        mol.basis = "cc-pvdz"
        mol.verbose = 0
        mol.build()

        mf = scf.RHF(mol)
        mf = mf.density_fit(auxbasis="cc-pv5z-ri")
        mf.conv_tol = 1e-11
        mf.kernel()

        mf.mo_coeff = mpi_helper.bcast_dict(mf.mo_coeff, root=0)
        mf.mo_energy = mpi_helper.bcast_dict(mf.mo_energy, root=0)

        agf2_pyscf = agf2.AGF2(mf)
        agf2_pyscf.max_cycle = 1
        agf2_pyscf.conv_tol = 1e-9
        agf2_pyscf.conv_tol_rdm1 = 1e-10
        agf2_pyscf.conv_tol_nelec = 1e-8
        agf2_pyscf.verbose = 5
        agf2_pyscf.kernel()

        cls.mol, cls.mf, cls.agf2_pyscf = mol, mf, agf2_pyscf

    @classmethod
    def tearDownClass(cls):
        del cls.mol, cls.mf, cls.agf2_pyscf

    def test_vs_pyscf(self):
        agf2 = GF2(self.mf)
        agf2.max_cycle = 1
        agf2.fock_loop = True
        agf2.fock_opts = dict(conv_tol_rdm1=1e-10, conv_tol_nelec=1e-8)
        agf2.conv_tol = 1e-9
        agf2.conv_tol_moms = 1e-6
        conv, gf, se, _ = agf2.kernel(nmom_max=1)
        gf_a = gf.physical(weight=1e-8)
        gf_b = self.agf2_pyscf.gf
        gf_b.remove_uncoupled(1e-8)
        self.assertAlmostEqual(
            gf_a.occupied().energies.max(),
            gf_b.get_occupied().energy.max(),
            4,
        )
        self.assertAlmostEqual(
            gf_a.virtual().energies.min(),
            gf_b.get_virtual().energy.min(),
            4,
        )

    def test_moments_vs_pyscf(self):
        gf2 = GF2(self.mf)
        th1, tp1 = gf2.build_se_moments(1, gf2.ao2mo())

        agf2_ref = agf2.AGF2(self.mf)
        gf = agf2_ref.init_gf()
        eri = agf2_ref.ao2mo()
        seh2 = agf2_ref.build_se_part(eri, gf.get_occupied(), gf.get_virtual())
        sep2 = agf2_ref.build_se_part(eri, gf.get_virtual(), gf.get_occupied())
        th2 = seh2.moment(range(2))
        tp2 = sep2.moment(range(2))

        np.testing.assert_allclose(th1, th2, rtol=np.inf, atol=1e-8)
        np.testing.assert_allclose(tp1, tp2, rtol=np.inf, atol=1e-8)

    def _test_regression(self, xc, kwargs, nmom_max, ip, ea, name=""):
        mol = gto.M(atom="H 0 0 0; Li 0 0 1.64", basis="6-31g", verbose=0)
        mf = scf.RHF(mol).density_fit().run()
        mf.mo_coeff = mpi_helper.bcast_dict(mf.mo_coeff, root=0)
        mf.mo_energy = mpi_helper.bcast_dict(mf.mo_energy, root=0)
        gf2 = GF2(mf, **kwargs)
        gf2.kernel(nmom_max)
        gf = gf2.gf.physical(weight=0.1)
        self.assertAlmostEqual(gf.occupied().energies[-1], ip, 7, msg=name)
        self.assertAlmostEqual(gf.virtual().energies[0], ea, 7, msg=name)

    def test_regression_simple(self):
        ip = -0.283356969966
        ea = 0.006080779020
        self._test_regression("hf", dict(), 1, ip, ea, "simple")

    def test_regression_fock_loop(self):
        ip = -0.281601986831
        ea = 0.005041215771
        self._test_regression("hf", dict(fock_loop=True), 3, ip, ea, "simple")


if __name__ == "__main__":
    print("Running tests for GF2")
    unittest.main()
