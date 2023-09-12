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


class Test_UGW_vs_RGW(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        mol = gto.Mole()
        mol.atom = "Li 0 0 0; H 0 0 1.64"
        mol.basis = "cc-pvdz"
        mol.verbose = 0
        mol.build()

        mf = dft.RKS(mol)
        mf.xc = "hf"
        mf.conv_tol = 1e-12
        mf = mf.density_fit()
        mf.with_df.build()
        mf.kernel()

        mf.mo_coeff = mpi_helper.bcast_dict(mf.mo_coeff, root=0)
        mf.mo_energy = mpi_helper.bcast_dict(mf.mo_energy, root=0)

        cls.mol, cls.mf = mol, mf

    @classmethod
    def tearDownClass(cls):
        del cls.mol, cls.mf

    def test_dtda(self):
        rgw = GW(self.mf)
        rgw.polarizability = "dtda"
        rgw.kernel(5)

        uhf = self.mf.to_uks()
        uhf.with_df = self.mf.with_df

        ugw = UGW(uhf)
        ugw.polarizability = "dtda"
        ugw.kernel(5)

        self.assertTrue(rgw.converged)
        self.assertTrue(ugw.converged)

        np.testing.assert_allclose(rgw.qp_energy, ugw.qp_energy[0])
        np.testing.assert_allclose(rgw.qp_energy, ugw.qp_energy[1])

    def test_drpa(self):
        rgw = GW(self.mf)
        rgw.polarizability = "drpa"
        rgw.kernel(5)

        uhf = self.mf.to_uks()
        uhf.with_df = self.mf.with_df

        ugw = UGW(uhf)
        ugw.polarizability = "drpa"
        ugw.kernel(5)

        self.assertTrue(rgw.converged)
        self.assertTrue(ugw.converged)

        np.testing.assert_allclose(rgw.qp_energy, ugw.qp_energy[0])
        np.testing.assert_allclose(rgw.qp_energy, ugw.qp_energy[1])


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

    def test_moments_vs_tdscf_rpa(self):
        if mpi_helper.size > 1:
            pytest.skip("Doesn't work with MPI")

        ugw = UGW(self.mf)
        ugw.diagonal_se = True
        ugw.compression = None
        nocc = ugw.nocc
        nvir = (ugw.nmo[0] - ugw.nocc[0], ugw.nmo[1] - ugw.nocc[1])
        th1, tp1 = ugw.build_se_moments(5, ugw.ao2mo())

        td = tdscf.dRPA(self.mf)
        td.nstates = max(nocc[0] * nvir[0], nocc[1] * nvir[1])
        td.kernel()
        z = (
            np.sum(np.array([x[0] for x in td.xy]) * 2, axis=1).reshape(len(td.e[0]), nocc[0] * nvir[0]),
            np.sum(np.array([x[1] for x in td.xy]) * 2, axis=1).reshape(len(td.e[1]), nocc[1] * nvir[1]),
        )
        integrals = ugw.ao2mo()
        Lpq = integrals.Lpx
        Lia = integrals.Lia

        m = (
            lib.einsum("Qx,vx,Qpj->vpj", Lia[0], z[0], Lpq[0][:, :, :nocc[0]]),
            lib.einsum("Qx,vx,Qpj->vpj", Lia[1], z[1], Lpq[1][:, :, :nocc[1]]),
        )
        e = (
            lib.direct_sum("j-v->jv", self.mf.mo_energy[0][:nocc[0]], td.e[0]),
            lib.direct_sum("j-v->jv", self.mf.mo_energy[1][:nocc[1]], td.e[1]),
        )
        th2 = []
        for n in range(6):
            t = (
                lib.einsum("vpj,jv,vqj->pq", m[0], np.power(e[0], n), m[0]),
                lib.einsum("vpj,jv,vqj->pq", m[1], np.power(e[1], n), m[1]),
            )
            if ugw.diagonal_se:
                t = (np.diag(np.diag(t[0])), np.diag(np.diag(t[1])))
            th2.append(t)

        m = (
            lib.einsum("Qx,vx,Qpj->vpj", Lia[0], z[0], Lpq[0][:, :, nocc[0]:]),
            lib.einsum("Qx,vx,Qpj->vpj", Lia[1], z[1], Lpq[1][:, :, nocc[1]:]),
        )
        e = (
            lib.direct_sum("j-v->jv", self.mf.mo_energy[0][nocc[0]:], td.e[0]),
            lib.direct_sum("j-v->jv", self.mf.mo_energy[1][nocc[1]:], td.e[1]),
        )
        tp2 = []
        for n in range(6):
            t = (
                lib.einsum("vpj,jv,vqj->pq", m[0], np.power(e[0], n), m[0]),
                lib.einsum("vpj,jv,vqj->pq", m[1], np.power(e[1], n), m[1]),
            )
            if ugw.diagonal_se:
                t = (np.diag(np.diag(t[0])), np.diag(np.diag(t[1])))
            tp2.append(t)

        for a, b in zip(th1, th2):
            dif = np.max(np.abs(a - b)) / np.max(np.abs(a))
            self.assertAlmostEqual(dif, 0, 8)
        for a, b in zip(tp1, tp2):
            dif = np.max(np.abs(a - b)) / np.max(np.abs(a))
            self.assertAlmostEqual(dif, 0, 8)

    def test_moments_vs_tdscf_tda(self):
        if mpi_helper.size > 1:
            pytest.skip("Doesn't work with MPI")

        ugw = UGW(self.mf)
        ugw.polarizability = "dtda"
        ugw.diagonal_se = True
        ugw.compression = None
        nocc = ugw.nocc
        nvir = (ugw.nmo[0] - ugw.nocc[0], ugw.nmo[1] - ugw.nocc[1])
        th1, tp1 = ugw.build_se_moments(5, ugw.ao2mo())

        td = tdscf.uhf.TDA(self.mf)
        td.nstates = max(nocc[0] * nvir[0], nocc[1] * nvir[1])
        td.kernel()
        xy = (
            np.array([x[0][0] for x in td.xy]).reshape(-1, nocc[0] * nvir[0]),
            np.array([x[0][1] for x in td.xy]).reshape(-1, nocc[1] * nvir[1]),
        )
        z = (xy[0] * 2, xy[1] * 2)
        integrals = ugw.ao2mo()
        Lpq = integrals.Lpx
        Lia = integrals.Lia
        print(z[0].shape, z[1].shape, td.e.shape)

        m = (
            lib.einsum("Qx,vx,Qpj->vpj", Lia[0], z[0], Lpq[0][:, :, :nocc[0]]),
            lib.einsum("Qx,vx,Qpj->vpj", Lia[1], z[1], Lpq[1][:, :, :nocc[1]]),
        )
        e = (
            lib.direct_sum("j-v->jv", self.mf.mo_energy[0][:nocc[0]], td.e[0]),
            lib.direct_sum("j-v->jv", self.mf.mo_energy[1][:nocc[1]], td.e[1]),
        )
        th2 = []
        for n in range(6):
            t = (
                lib.einsum("vpj,jv,vqj->pq", m[0], np.power(e[0], n), m[0]),
                lib.einsum("vpj,jv,vqj->pq", m[1], np.power(e[1], n), m[1]),
            )
            if ugw.diagonal_se:
                t = (np.diag(np.diag(t[0])), np.diag(np.diag(t[1])))
            th2.append(t)

        m = (
            lib.einsum("Qx,vx,Qpj->vpj", Lia[0], z[0], Lpq[0][:, :, nocc[0]:]),
            lib.einsum("Qx,vx,Qpj->vpj", Lia[1], z[1], Lpq[1][:, :, nocc[1]:]),
        )
        e = (
            lib.direct_sum("j-v->jv", self.mf.mo_energy[0][nocc[0]:], td.e[0]),
            lib.direct_sum("j-v->jv", self.mf.mo_energy[1][nocc[1]:], td.e[1]),
        )
        tp2 = []
        for n in range(6):
            t = (
                lib.einsum("vpj,jv,vqj->pq", m[0], np.power(e[0], n), m[0]),
                lib.einsum("vpj,jv,vqj->pq", m[1], np.power(e[1], n), m[1]),
            )
            if ugw.diagonal_se:
                t = (np.diag(np.diag(t[0])), np.diag(np.diag(t[1])))
            tp2.append(t)

        for a, b in zip(th1, th2):
            dif = np.max(np.abs(a - b)) / np.max(np.abs(a))
            self.assertAlmostEqual(dif, 0, 8)
        for a, b in zip(tp1, tp2):
            dif = np.max(np.abs(a - b)) / np.max(np.abs(a))
            self.assertAlmostEqual(dif, 0, 8)

    def test_vs_pyscf_rpa(self):
        ugw = UGW(self.mf)
        ugw.diagonal_se = True
        ugw.compression = None
        ugw.npoints = 64
        conv, gf, se, _ = ugw.kernel(nmom_max=9)
        ugw_exact = gw.ugw_ac.UGWAC(self.mf)
        ugw_exact.kernel()
        print(ugw.qp_energy[0])
        print(ugw_exact.mo_energy[0])
        print(ugw.qp_energy[1])
        print(ugw_exact.mo_energy[1])
        self.assertAlmostEqual(
            ugw.qp_energy[0][ugw_exact.mo_occ[0] > 0].max(),
            ugw_exact.mo_energy[0][ugw_exact.mo_occ[0] > 0].max(),
            2,
        )
        self.assertAlmostEqual(
            ugw.qp_energy[0][ugw_exact.mo_occ[0] == 0].min(),
            ugw_exact.mo_energy[0][ugw_exact.mo_occ[0] == 0].min(),
            2,
        )
        self.assertAlmostEqual(
            ugw.qp_energy[1][ugw_exact.mo_occ[1] > 1].max(),
            ugw_exact.mo_energy[1][ugw_exact.mo_occ[1] > 1].max(),
            2,
        )
        self.assertAlmostEqual(
            ugw.qp_energy[1][ugw_exact.mo_occ[1] == 1].min(),
            ugw_exact.mo_energy[1][ugw_exact.mo_occ[1] == 1].min(),
            2,
        )


if __name__ == "__main__":
    print("Running tests for UGW")
    unittest.main()
