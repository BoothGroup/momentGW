"""
Tests for `scgw.py`.
"""

import unittest

from pyscf import dft, gto
from pyscf.agf2 import mpi_helper

from momentGW import scGW


class Test_scGW(unittest.TestCase):
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

    def test_nelec(self):
        gw = scGW(self.mf)
        gw.diagonal_se = True
        gw.optimise_chempot = True
        conv, gf, se, _ = gw.kernel(nmom_max=1)
        self.assertAlmostEqual(
            gf.occupied().moment(0).trace() * 2,
            self.mol.nelectron,
            8,
        )
        gw.fock_loop = True
        conv, gf, se, _ = gw.kernel(nmom_max=1)
        self.assertAlmostEqual(
            gf.occupied().moment(0).trace() * 2,
            self.mol.nelectron,
            8,
        )

    def _test_regression(self, xc, kwargs, nmom_max, ip, ea, name=""):
        mol = gto.M(atom="H 0 0 0; Li 0 0 1.64", basis="6-31g", verbose=0)
        mf = dft.RKS(mol, xc=xc).density_fit().run()
        mf.mo_coeff = mpi_helper.bcast_dict(mf.mo_coeff, root=0)
        mf.mo_energy = mpi_helper.bcast_dict(mf.mo_energy, root=0)
        gw = scGW(mf, **kwargs)
        gw.conv_tol = 1e-9
        gw.max_cycle = 200
        gw.kernel(nmom_max)
        gf = gw.gf.physical(weight=0.1)
        self.assertTrue(gw.converged)
        self.assertAlmostEqual(gf.occupied().energies[-1], ip, 7, msg=name)
        self.assertAlmostEqual(gf.virtual().energies[0], ea, 7, msg=name)

    def test_regression_simple(self):
        ip = -0.2812207212
        ea = 0.0076809093
        self._test_regression("hf", dict(), 1, ip, ea, "simple")

    def test_regression_gw0(self):
        ip = -0.2790224850
        ea = 0.0068765373
        self._test_regression("hf", dict(w0=True), 3, ip, ea, "gw0")

    def test_regression_g0w(self):
        ip = -0.2792300309
        ea = 0.0077792812
        self._test_regression("hf", dict(g0=True, damping=0.5), 1, ip, ea, "g0w")

    def test_regression_pbe_fock_loop(self):
        ip = -0.2865782158
        ea = 0.0062598632
        self._test_regression("pbe", dict(fock_loop=True), 1, ip, ea, "pbe fock loop")

    def test_regression_frozen_fock_loop(self):
        ip = -0.2768253593
        ea = 0.0060856962
        self._test_regression("hf", dict(fock_loop=True, frozen=[-2, -1]), 1, ip, ea, "frozen fock loop")

    def test_regression_rdm(self):
        ip = -0.2826880460
        ea = 0.0065479379
        self._test_regression("hf", dict(rdm_correction=True), 1, ip, ea, "rdm correction")

    def test_regression_pbe_rdm(self):
        ip = -0.2826880466
        ea = 0.0065479399
        self._test_regression("pbe", dict(rdm_correction=True), 1, ip, ea, "pbe rdm correction")


if __name__ == "__main__":
    print("Running tests for scGW")
    unittest.main()
