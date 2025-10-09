"""
Tests for `evgw.py`.
"""

import unittest

from pyscf import dft, gto
from pyscf.agf2 import mpi_helper

from momentGW import evGW


class Test_evGW(unittest.TestCase):
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
        gw = evGW(self.mf)
        gw.diagonal_se = True
        conv, gf, se, _ = gw.kernel(nmom_max=1)
        self.assertAlmostEqual(
            gf.occupied().moment(0).trace() * 2,
            self.mol.nelectron,
            1,
        )
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
        mf = dft.RKS(mol, xc=xc).density_fit(auxbasis='weigend').run()
        mf.mo_coeff = mpi_helper.bcast_dict(mf.mo_coeff, root=0)
        mf.mo_energy = mpi_helper.bcast_dict(mf.mo_energy, root=0)
        gw = evGW(mf, **kwargs)
        gw.max_cycle = 250
        gw.conv_tol_moms = 1e-4
        gw.conv_tol = 1e-8
        gw.kernel(nmom_max)
        gf = gw.gf.physical(weight=0.1)
        self.assertTrue(gw.converged)
        self.assertAlmostEqual(gf.occupied().energies[-1], ip, 6, msg=name)
        self.assertAlmostEqual(gf.virtual().energies[0], ea, 6, msg=name)

    def test_regression_simple(self):
        ip = -0.2786687811494432
        ea = 0.006158900147976394
        self._test_regression("hf", dict(), 1, ip, ea, "simple")

    def test_regression_gw0(self):
        ip = -0.27655204792484694
        ea = 0.005525397394672053
        self._test_regression("hf", dict(w0=True), 3, ip, ea, "gw0")

    def test_regression_g0w(self):
        ip = -0.27935804486799726
        ea = 0.006156582803460838
        self._test_regression("hf", dict(g0=True, damping=0.5), 1, ip, ea, "g0w")

    def test_regression_pbe_fock_loop(self):
        ip = -0.28141299447625784
        ea = 0.00726764167546422
        self._test_regression("pbe", dict(), 1, ip, ea, "pbe")


if __name__ == "__main__":
    print("Running tests for evGW")
    unittest.main()
