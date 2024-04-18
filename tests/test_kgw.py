"""
Tests for `pbc/gw.py`
"""

import unittest

import numpy as np
from pyscf.agf2 import mpi_helper
from pyscf.pbc import dft, gto
from pyscf.pbc.tools import k2gamma

from momentGW import GW, KGW


class Test_KGW(unittest.TestCase):
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
        # mf = scf.KRHF(cell, kpts)
        mf = mf.density_fit(auxbasis="weigend")
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
        smf.with_df.force_dm_kbuild = True

        cls.cell, cls.kpts, cls.mf, cls.smf = cell, kpts, mf, smf

    @classmethod
    def tearDownClass(cls):
        del cls.cell, cls.kpts, cls.mf, cls.smf

    def test_supercell_valid(self):
        # Require real MOs for supercell comparison

        scell, phase = k2gamma.get_phase(self.cell, self.kpts)
        nk, nao, nmo = np.shape(self.mf.mo_coeff)
        nr, _ = np.shape(phase)

        k_conj_groups = k2gamma.group_by_conj_pairs(self.cell, self.kpts, return_kpts_pairs=False)
        k_phase = np.eye(nk, dtype=np.complex128)
        r2x2 = np.array([[1.0, 1j], [1.0, -1j]]) * 0.5**0.5
        pairs = [[k, k_conj] for k, k_conj in k_conj_groups if k_conj is not None and k != k_conj]
        for idx in np.array(pairs):
            k_phase[idx[:, None], idx] = r2x2

        c_gamma = np.einsum("Rk,kum,kh->Ruhm", phase, self.mf.mo_coeff, k_phase)
        c_gamma = c_gamma.reshape(nao * nr, nk * nmo)
        c_gamma[:, abs(c_gamma.real).max(axis=0) < 1e-5] *= -1j

        self.assertAlmostEqual(np.max(np.abs(np.array(c_gamma).imag)), 0, 8)

    def _test_vs_supercell(self, gw, kgw, full=False, tol=1e-8):
        e1 = np.concatenate([gf.energies for gf in kgw.gf])
        w1 = np.concatenate([gf.weights() for gf in kgw.gf])
        mask = np.argsort(e1)
        e1 = e1[mask]
        w1 = w1[mask]
        e2 = gw.gf.energies
        w2 = gw.gf.weights()
        if full:
            np.testing.assert_allclose(e1, e2, atol=tol)
        else:
            np.testing.assert_allclose(e1[w1 > 1e-1], e2[w2 > 1e-1], atol=tol)

    def test_dtda_vs_supercell(self):
        nmom_max = 5

        kgw = KGW(self.mf)
        kgw.polarizability = "dtda"
        kgw.kernel(nmom_max)

        gw = GW(self.smf)
        gw.__dict__.update({opt: getattr(kgw, opt) for opt in kgw._opts})
        gw.kernel(nmom_max)

        self._test_vs_supercell(gw, kgw, full=True)

    def test_drpa_vs_supercell(self):
        nmom_max = 5

        kgw = KGW(self.mf)
        kgw.polarizability = "drpa"
        kgw.kernel(nmom_max)

        gw = GW(self.smf)
        gw.__dict__.update({opt: getattr(kgw, opt) for opt in kgw._opts})
        gw.kernel(nmom_max)

        self._test_vs_supercell(gw, kgw, full=True)

    def test_dtda_vs_supercell_fock_loop(self):
        nmom_max = 5

        kgw = KGW(self.mf)
        kgw.polarizability = "dtda"
        kgw.fock_loop = True
        kgw.compression = None
        kgw.kernel(nmom_max)

        gw = GW(self.smf)
        gw.__dict__.update({opt: getattr(kgw, opt) for opt in kgw._opts})
        gw.kernel(nmom_max)

        self._test_vs_supercell(gw, kgw)

    def test_drpa_vs_supercell_fock_loop(self):
        nmom_max = 5

        kgw = KGW(self.mf)
        kgw.polarizability = "drpa"
        kgw.fock_loop = True
        kgw.compression = None
        kgw.kernel(nmom_max)

        gw = GW(self.smf)
        gw.__dict__.update({opt: getattr(kgw, opt) for opt in kgw._opts})
        gw.kernel(nmom_max)

        self._test_vs_supercell(gw, kgw)

    def test_dtda_vs_supercell_compression(self):
        nmom_max = 5

        kgw = KGW(self.mf)
        kgw.polarizability = "dtda"
        kgw.compression = "ov,oo,vv"
        kgw.compression_tol = 1e-7
        kgw.kernel(nmom_max)

        gw = GW(self.smf)
        gw.__dict__.update({opt: getattr(kgw, opt) for opt in kgw._opts})
        gw.kernel(nmom_max)

        self._test_vs_supercell(gw, kgw, full=False, tol=1e-5)

    def test_drpa_vs_supercell_compression(self):
        nmom_max = 5

        kgw = KGW(self.mf)
        kgw.polarizability = "drpa"
        kgw.compression = "ov,oo,vv"
        kgw.compression_tol = 1e-7
        kgw.kernel(nmom_max)

        gw = GW(self.smf)
        gw.__dict__.update({opt: getattr(kgw, opt) for opt in kgw._opts})
        gw.kernel(nmom_max)

        self._test_vs_supercell(gw, kgw, full=False, tol=1e-5)

    def test_dtda_vs_supercell_frozen(self):
        nmom_max = 3

        kgw = KGW(self.mf)
        kgw.polarizability = "dtda"
        kgw.frozen = [0]
        kgw.kernel(nmom_max)

        gw = GW(self.smf)
        gw.__dict__.update({opt: getattr(kgw, opt) for opt in kgw._opts})
        gw.frozen = list(range(len(self.kpts)))
        gw.kernel(nmom_max)

        self._test_vs_supercell(gw, kgw, full=True)


if __name__ == "__main__":
    print("Running tests for KGW")
    unittest.main()

class Test_unit_KGW(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cell = gto.Cell()
        cell.atom = "He 0 0 0; He 1 1 1"
        cell.basis = "6-31g"
        cell.a = np.eye(3) * 3
        cell.max_memory = 1e10
        cell.verbose = 0
        cell.build()

        kmesh = [1, 1, 1]
        kpts = cell.make_kpts(kmesh)

        mf = dft.KRKS(cell, kpts, xc="hf")
        # mf = scf.KRHF(cell, kpts)
        mf = mf.density_fit(auxbasis="weigend")
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
        smf.with_df.force_dm_kbuild = True

        cls.cell, cls.kpts, cls.mf, cls.smf = cell, kpts, mf, smf

    @classmethod
    def tearDownClass(cls):
        del cls.cell, cls.kpts, cls.mf, cls.smf

    def _test_vs_supercell(self, gw, kgw, full=False, tol=1e-8):
        e1 = np.concatenate([gf.energies for gf in kgw.gf])
        w1 = np.concatenate([gf.weights() for gf in kgw.gf])
        mask = np.argsort(e1)
        e1 = e1[mask]
        w1 = w1[mask]
        e2 = gw.gf.energies
        w2 = gw.gf.weights()
        if full:
            np.testing.assert_allclose(e1, e2, atol=tol)
        else:
            np.testing.assert_allclose(e1[w1 > 1e-1], e2[w2 > 1e-1], atol=tol)

    def test_dtda_vs_supercell(self):
        nmom_max = 5

        kgw = KGW(self.mf)
        kgw.polarizability = "dtda"
        kgw.kernel(nmom_max)

        gw = GW(self.smf)
        gw.__dict__.update({opt: getattr(kgw, opt) for opt in kgw._opts})
        gw.kernel(nmom_max)

        self._test_vs_supercell(gw, kgw, full=False)

    def test_drpa_vs_supercell(self):
        nmom_max = 5

        kgw = KGW(self.mf)
        kgw.polarizability = "drpa"
        kgw.kernel(nmom_max)

        gw = GW(self.smf)
        gw.__dict__.update({opt: getattr(kgw, opt) for opt in kgw._opts})
        gw.kernel(nmom_max)

        self._test_vs_supercell(gw, kgw, full=False)

if __name__ == "__main__":
    print("Running unit tests for KGW")
    unittest.main()