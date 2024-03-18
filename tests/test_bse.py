"""
Tests for `bse.py`
"""

import unittest

import numpy as np
import pytest
from pyscf import dft, gto, gw, lib, tdscf
from pyscf.agf2 import mpi_helper

from momentGW import BSE, GW


class Test_BSE(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        mol = gto.Mole()
        mol.atom = "Li 0 0 0; H 0 0 1.64"
        mol.basis = "cc-pvdz"
        mol.verbose = 0
        mol.build()

        mf = dft.RKS(mol)
        mf = mf.density_fit()
        mf.xc = "hf"
        mf.conv_tol = 1e-11
        mf.kernel()

        mf.mo_coeff = mpi_helper.bcast_dict(mf.mo_coeff, root=0)
        mf.mo_energy = mpi_helper.bcast_dict(mf.mo_energy, root=0)

        cls.mol, cls.mf = mol, mf

    def test_tda_bse_singlet(self):
        nmom_gw = 9
        nmom_bse = 13

        gw = GW(self.mf, polarizability="dtda", compression=None)
        gw.kernel(nmom_gw)

        bse = BSE(gw, excitation="singlet")
        gf = bse.kernel(nmom_bse)

        # Reference values from Bintrim & Berkelbach
        # Low precision because of slow convergence of moment BSE
        self.assertAlmostEqual(gf.energies[0], 0.13234222, 2)
        self.assertAlmostEqual(gf.energies[1], 0.17192952, 2)
        self.assertAlmostEqual(gf.energies[2], 0.17192952, 2)

    def test_tda_bse_triplet(self):
        nmom_gw = 9
        nmom_bse = 13

        gw = GW(self.mf, polarizability="dtda", compression=None)
        gw.kernel(nmom_gw)

        bse = BSE(gw, excitation="triplet")
        gf = bse.kernel(nmom_bse)

        # Reference values from Bintrim & Berkelbach
        # Low precision because of slow convergence of moment BSE
        self.assertAlmostEqual(gf.energies[0], 0.10655911, 2)
        self.assertAlmostEqual(gf.energies[1], 0.14420611, 2)
        self.assertAlmostEqual(gf.energies[2], 0.14420611, 2)


if __name__ == "__main__":
    print("Running tests for BSE")
    unittest.main()
