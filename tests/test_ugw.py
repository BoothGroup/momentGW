"""
Tests for `uhf/gw.py`.
"""

import unittest

import numpy as np
import pytest
from pyscf import dft, gto, gw, lib, tdscf
from pyscf.agf2 import mpi_helper

from momentGW import GW
from momentGW.uhf import UGW


class Test_UGW(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        mol = gto.Mole()
        mol.atom = "Be 0 0 0; H 0 0 1"
        mol.basis = "cc-pvdz"
        mol.spin = 1
        mol.verbose = 0
        mol.build()

        mf = dft.UKS(mol)
        mf.xc = "hf"
        mf.conv_tol = 1e-12
        mf.kernel()

        mf.mo_coeff = (
            mpi_helper.bcast_dict(mf.mo_coeff[0], root=0),
            mpi_helper.bcast_dict(mf.mo_coeff[1], root=0),
        )
        mf.mo_energy = (
            mpi_helper.bcast_dict(mf.mo_energy[0], root=0),
            mpi_helper.bcast_dict(mf.mo_energy[1], root=0),
        )

        mf = mf.density_fit(auxbasis="cc-pv5z-ri")
        mf.with_df.build()

        cls.mol, cls.mf = mol, mf

    @classmethod
    def tearDownClass(cls):
        del cls.mol, cls.mf

    def test_vs_pyscf_vhf_df(self):
        ugw = UGW(self.mf)
        ugw.polarizability = "dtda"
        ugw.compression = None
        ugw.diagonal_se = True
        conv, gf, se, _ = ugw.kernel(nmom_max=7)
        gf[0].remove_uncoupled(tol=1e-8)
        gf[1].remove_uncoupled(tol=1e-8)

        gw_exact = gw.ugw_ac.UGWAC(self.mf, tdmf=self.mf.dTDA())
        gw_exact.kernel()

        self.assertAlmostEqual(
            gf.get_occupied().energy.max(),
            gw_exact.mo_energy[gw_exact.mo_occ > 0].max(),
            2,
        )
        self.assertAlmostEqual(
            gf.get_virtual().energy.min(),
            gw_exact.mo_energy[gw_exact.mo_occ == 0].min(),
            2,
        )


if __name__ == "__main__":
    print("Running tests for UGW")
    unittest.main()
