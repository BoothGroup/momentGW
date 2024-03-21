"""
Tests for `fsgw.py`.
"""

import unittest

import numpy as np
from pyscf import dft, gto
from pyscf.agf2 import mpi_helper

from momentGW import fsGW


class Test_fsGW(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pass

    @classmethod
    def tearDownClass(cls):
        pass

    def _test_regression(self, xc, kwargs, nmom_max, ip, ea, ip_full, ea_full, name=""):
        mol = gto.M(atom="H 0 0 0; Li 0 0 1.64", basis="6-31g", verbose=0)
        mf = dft.RKS(mol, xc=xc).density_fit().run()
        mf.mo_coeff = mpi_helper.bcast_dict(mf.mo_coeff, root=0)
        mf.mo_energy = mpi_helper.bcast_dict(mf.mo_energy, root=0)
        gw = fsGW(mf, **kwargs)
        gw.max_cycle = 200
        gw.conv_tol = 1e-8
        gw.conv_tol_qp = 1e-8
        gw.damping = 0.5
        gw.kernel(nmom_max)
        gf = gw.gf.physical(weight=0.5)
        qp_energy = gw.qp_energy
        self.assertTrue(gw.converged)
        self.assertAlmostEqual(np.max(qp_energy[mf.mo_occ > 0]), ip, 6, msg=name)
        self.assertAlmostEqual(np.min(qp_energy[mf.mo_occ == 0]), ea, 6, msg=name)
        self.assertAlmostEqual(gf.occupied().energies[-1], ip_full, 6, msg=name)
        self.assertAlmostEqual(gf.virtual().energies[0], ea_full, 6, msg=name)

    def test_regression_simple(self):
        # Quasiparticle energies:
        ip = -0.298368381748
        ea = 0.009241657362
        # GF poles:
        ip_full = -0.285192929632
        ea_full = 0.006413219011
        self._test_regression("hf", dict(), 1, ip, ea, ip_full, ea_full, "simple")

    def test_regression_pbe(self):
        # Quasiparticle energies:
        ip = -0.298631888610
        ea = 0.009232004612
        # GF poles:
        ip_full = -0.279446170582
        ea_full = 0.006189745568
        self._test_regression("pbe", dict(), 3, ip, ea, ip_full, ea_full, "pbe srg")


if __name__ == "__main__":
    print("Running tests for fsGW")
    unittest.main()
