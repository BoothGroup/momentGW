"""
Tests for `pbc/uhf/gw.py`
"""

import unittest

import numpy as np
from pyscf import lib
from pyscf.agf2 import mpi_helper
from pyscf.pbc import dft, gto
from pyscf.pbc.tools import k2gamma

from momentGW import GW, KGW, UGW, KUGW


class Test_KUGW_vs_KRGW(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cell = gto.Cell()
        cell.atom = "He 0 0 0; He 1 1 1"
        cell.basis = "6-31g"
        cell.a = np.eye(3) * 3
        cell.max_memory = 1e10
        cell.verbose = 0
        cell.build()

        kmesh = [3, 1, 1]
        kpts = cell.make_kpts(kmesh)

        mf = dft.KRKS(cell, kpts, xc="hf")
        mf = mf.density_fit(auxbasis="weigend")
        mf.with_df._prefer_ccdf = True
        mf.with_df.force_dm_kbuild = True
        mf.exxdiv = None
        mf.conv_tol = 1e-10
        mf.kernel()

        for k in range(len(kpts)):
            mf.mo_coeff[k] = mpi_helper.bcast_dict(mf.mo_coeff[k], root=0)
            mf.mo_energy[k] = mpi_helper.bcast_dict(mf.mo_energy[k], root=0)

        smf = k2gamma.k2gamma(mf, kmesh=kmesh)
        smf = smf.density_fit(auxbasis="weigend")
        smf.exxdiv = None
        smf.with_df._prefer_ccdf = True
        smf.with_df.force_dm_kbuild = True

        cls.cell, cls.kpts, cls.mf, cls.smf = cell, kpts, mf, smf

    @classmethod
    def tearDownClass(cls):
        del cls.cell, cls.kpts, cls.mf, cls.smf

    def test_dtda(self):
        krgw = KGW(self.mf)
        krgw.compression = None
        krgw.polarizability = "dtda"
        krgw.kernel(3)

        uhf = self.mf.to_uhf()
        uhf.with_df = self.mf.with_df

        kugw = KUGW(uhf)
        kugw.compression = None
        kugw.polarizability = "dtda"
        kugw.kernel(3)

        self.assertTrue(krgw.converged)
        self.assertTrue(kugw.converged)

        np.testing.assert_allclose(krgw.qp_energy, kugw.qp_energy[0])
        np.testing.assert_allclose(krgw.qp_energy, kugw.qp_energy[1])

    #def test_dtda_compression(self):
    #    krgw = KGW(self.mf)
    #    krgw.compression = "ov,oo"
    #    krgw.compression_tol = 1e-4
    #    krgw.polarizability = "dtda"
    #    krgw.kernel(3)

    #    uhf = self.mf.to_uhf()
    #    uhf.with_df = self.mf.with_df

    #    kugw = KUGW(uhf)
    #    kugw.compression = "ov,oo"
    #    kugw.compression_tol = 1e-4
    #    kugw.polarizability = "dtda"
    #    kugw.kernel(3)

    #    self.assertTrue(krgw.converged)
    #    self.assertTrue(kugw.converged)

    #    np.testing.assert_allclose(krgw.qp_energy, kugw.qp_energy[0])
    #    np.testing.assert_allclose(krgw.qp_energy, kugw.qp_energy[1])


class Test_KUGW(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cell = gto.Cell()
        cell.atom = "Be 0 0 0; H 1 1 1"
        cell.basis = "6-31g"
        cell.spin = 1
        cell.a = np.eye(3) * 3
        cell.max_memory = 1e10
        cell.verbose = 0
        cell.build()

        kmesh = [3, 1, 1]
        kpts = cell.make_kpts(kmesh)

        mf = dft.KUKS(cell, kpts, xc="hf")
        mf = mf.density_fit(auxbasis="weigend")
        mf.with_df._prefer_ccdf = True
        mf.with_df.force_dm_kbuild = True
        mf.exxdiv = None
        mf.conv_tol = 1e-10
        mf.kernel()

        for s in range(2):
            for k in range(len(kpts)):
                mf.mo_coeff[s][k] = mpi_helper.bcast_dict(mf.mo_coeff[s][k], root=0)
                mf.mo_energy[s][k] = mpi_helper.bcast_dict(mf.mo_energy[s][k], root=0)

        smf = k2gamma.k2gamma(mf, kmesh=kmesh)
        smf = smf.density_fit(auxbasis="weigend")
        smf.exxdiv = None
        smf.with_df._prefer_ccdf = True
        smf.with_df.force_dm_kbuild = True

        cls.cell, cls.kpts, cls.mf, cls.smf = cell, kpts, mf, smf

    @classmethod
    def tearDownClass(cls):
        del cls.cell, cls.kpts, cls.mf, cls.smf

    # TODO adapt for UKS
    #def test_supercell_valid(self):
    #    # Require real MOs for supercell comparison

    #    scell, phase = k2gamma.get_phase(self.cell, self.kpts)
    #    nk, nao, nmo = np.shape(self.mf.mo_coeff)
    #    nr, _ = np.shape(phase)

    #    k_conj_groups = k2gamma.group_by_conj_pairs(self.cell, self.kpts, return_kpts_pairs=False)
    #    k_phase = np.eye(nk, dtype=np.complex128)
    #    r2x2 = np.array([[1., 1j], [1., -1j]]) * .5**.5
    #    pairs = [[k, k_conj] for k, k_conj in k_conj_groups
    #             if k_conj is not None and k != k_conj]
    #    for idx in np.array(pairs):
    #        k_phase[idx[:, None], idx] = r2x2

    #    c_gamma = np.einsum('Rk,kum,kh->Ruhm', phase, self.mf.mo_coeff, k_phase)
    #    c_gamma = c_gamma.reshape(nao*nr, nk*nmo)
    #    c_gamma[:, abs(c_gamma.real).max(axis=0) < 1e-5] *= -1j

    #    self.assertAlmostEqual(np.max(np.abs(np.array(c_gamma).imag)), 0, 8)

    #def _test_vs_supercell(self, gw, kgw, full=False, tol=1e-8):
    #    e1 = np.concatenate([gf.energy for gf in kgw.gf])
    #    w1 = np.concatenate([np.linalg.norm(gf.coupling, axis=0)**2 for gf in kgw.gf])
    #    mask = np.argsort(e1)
    #    e1 = e1[mask]
    #    w1 = w1[mask]
    #    e2 = gw.gf.energy
    #    w2 = np.linalg.norm(gw.gf.coupling, axis=0)**2
    #    if full:
    #        np.testing.assert_allclose(e1, e2, atol=tol)
    #    else:
    #        np.testing.assert_allclose(e1[w1 > 1e-1], e2[w2 > 1e-1], atol=tol)

    #def test_dtda_vs_supercell(self):
    #    nmom_max = 5

    #    kgw = KGW(self.mf)
    #    kgw.polarizability = "dtda"
    #    kgw.kernel(nmom_max)

    #    gw = GW(self.smf)
    #    gw.__dict__.update({opt: getattr(kgw, opt) for opt in kgw._opts})
    #    gw.kernel(nmom_max)

    #    self._test_vs_supercell(gw, kgw, full=True)


class Test_KUGW_no_beta(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cell = gto.Cell()
        cell.atom = "H 0 0 0; H 1 1 1"
        cell.basis = "sto3g"
        cell.charge = 1
        cell.spin = 1
        cell.a = np.eye(3) * 3
        cell.max_memory = 1e10
        cell.verbose = 0
        cell.build()

        kmesh = [3, 1, 1]
        kpts = cell.make_kpts(kmesh)

        mf = dft.KUKS(cell, kpts, xc="hf")
        mf = mf.density_fit(auxbasis="weigend")
        mf.with_df._prefer_ccdf = True
        mf.with_df.force_dm_kbuild = True
        mf.exxdiv = None
        mf.conv_tol = 1e-10
        mf.kernel()

        for s in range(2):
            for k in range(len(kpts)):
                mf.mo_coeff[s][k] = mpi_helper.bcast_dict(mf.mo_coeff[s][k], root=0)
                mf.mo_energy[s][k] = mpi_helper.bcast_dict(mf.mo_energy[s][k], root=0)

        cls.cell, cls.kpts, cls.mf = cell, kpts, mf

    @classmethod
    def tearDownClass(cls):
        del cls.cell, cls.kpts, cls.mf

    def test_dtda_regression(self):
        kugw = KUGW(self.mf)
        kugw.compression = None
        kugw.polarizability = "dtda"
        kugw.kernel(3)

        self.assertTrue(kugw.converged)

        self.assertAlmostEqual(lib.fp(kugw.qp_energy[0]), -0.0042127651)
        self.assertAlmostEqual(lib.fp(kugw.qp_energy[1]), -0.0785013870)


if __name__ == "__main__":
    print("Running tests for KUGW")
    unittest.main()
