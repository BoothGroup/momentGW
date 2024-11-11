"""
Tests for `qsgw.py`.
"""

import unittest

import numpy as np
from pyscf import dft, gto
from pyscf.agf2 import mpi_helper

from momentGW import qsGW


class Test_qsGW(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pass

    @classmethod
    def tearDownClass(cls):
        pass

    def _test_regression(self, xc, kwargs, nmom_max, ip, ea, ip_full, ea_full, name=""):
        mol = gto.M(atom="H 0 0 0; Li 0 0 1.64", basis="sto3g", verbose=0)
        mf = dft.RKS(mol, xc=xc).density_fit().run()
        mf.mo_coeff = mpi_helper.bcast_dict(mf.mo_coeff, root=0)
        mf.mo_energy = mpi_helper.bcast_dict(mf.mo_energy, root=0)
        gw = qsGW(mf, **kwargs)
        gw.max_cycle = 200
        gw.conv_tol = 1e-8
        gw.conv_tol_qp = 1e-8
        gw.kernel(nmom_max)
        gf = gw.gf.physical(weight=0.5)
        qp_energy = gw.qp_energy
        self.assertTrue(gw.converged)
        self.assertAlmostEqual(np.max(qp_energy[gw.mo_occ > 0]), ip, 6, msg=name)
        self.assertAlmostEqual(np.min(qp_energy[gw.mo_occ == 0]), ea, 6, msg=name)
        self.assertAlmostEqual(gf.occupied().energies[-1], ip_full, 5, msg=name)
        self.assertAlmostEqual(gf.virtual().energies[0], ea_full, 5, msg=name)

    def test_regression_simple(self):
        # Quasiparticle energies:
        ip = -0.274195084690
        ea = 0.076494883697
        # GF poles:
        ip_full = -0.264403349572
        ea_full = 0.074970718795
        self._test_regression("hf", dict(), 1, ip, ea, ip_full, ea_full, "simple")

    def test_regression_pbe_srg(self):
        # Quasiparticle energies:
        ip = -0.271017148443
        ea = 0.075779760938
        # GF poles:
        ip_full = -0.39350150932022504
        ea_full = 0.170103953696
        self._test_regression("pbe", dict(srg=1000), 3, ip, ea, ip_full, ea_full, "pbe srg")

    def test_regression_frozen(self):
        # Quasiparticle energies:
        ip = -0.273897469018
        ea = 0.076209904753
        # GF poles:
        ip_full = -0.261855837372
        ea_full = 0.074140057899
        self._test_regression("hf", dict(frozen=[-1]), 3, ip, ea, ip_full, ea_full, "frozen")


if __name__ == "__main__":
    print("Running tests for qsGW")
    unittest.main()
