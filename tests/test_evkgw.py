"""
Tests for `pbc/evgw.py`
"""

import unittest

import numpy as np
from pyscf.agf2 import mpi_helper
from pyscf.pbc import dft, gto
from pyscf.pbc.tools import k2gamma

from momentGW import evGW, evKGW


class Test_evKGW(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cell = gto.Cell()
        cell.atom = "He 0 0 0; He 1 1 1"
        cell.basis = "6-31g"
        cell.a = np.eye(3) * 3
        cell.verbose = 0
        cell.build()

        kmesh = [3, 1, 1]
        kpts = cell.make_kpts(kmesh)

        mf = dft.KRKS(cell, kpts, xc="hf")
        mf = mf.density_fit(auxbasis="weigend")
        mf.conv_tol = 1e-10
        mf.kernel()

        for k in range(len(kpts)):
            mf.mo_coeff[k] = mpi_helper.bcast_dict(mf.mo_coeff[k], root=0)
            mf.mo_energy[k] = mpi_helper.bcast_dict(mf.mo_energy[k], root=0)

        smf = k2gamma.k2gamma(mf, kmesh=kmesh)
        smf = smf.density_fit(auxbasis="weigend")

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
        r2x2 = np.array([[1., 1j], [1., -1j]]) * .5**.5
        pairs = [[k, k_conj] for k, k_conj in k_conj_groups
                 if k_conj is not None and k != k_conj]
        for idx in np.array(pairs):
            k_phase[idx[:, None], idx] = r2x2

        c_gamma = np.einsum('Rk,kum,kh->Ruhm', phase, self.mf.mo_coeff, k_phase)
        c_gamma = c_gamma.reshape(nao*nr, nk*nmo)
        c_gamma[:, abs(c_gamma.real).max(axis=0) < 1e-5] *= -1j

        self.assertAlmostEqual(np.max(np.abs(np.array(c_gamma).imag)), 0, 8)

    def _test_vs_supercell(self, gw, kgw, full=False, check_convergence=True):
        if check_convergence:
            self.assertTrue(gw.converged)
            self.assertTrue(kgw.converged)
        e1 = np.concatenate([gf.energies for gf in kgw.gf])
        w1 = np.concatenate([gf.weights() for gf in kgw.gf])
        mask = np.argsort(e1)
        e1 = e1[mask]
        w1 = w1[mask]
        e2 = gw.gf.energies
        w2 = gw.gf.weights()
        if full:
            np.testing.assert_allclose(e1, e2, atol=1e-8)
        else:
            np.testing.assert_allclose(e1[w1 > 0.5], e2[w2 > 0.5], atol=1e-8)

    def test_dtda_vs_supercell(self):
        nmom_max = 3

        kgw = evKGW(self.mf)
        kgw.polarizability = "dtda"
        kgw.max_cycle = 50
        kgw.conv_tol = 1e-8
        kgw.damping = 0.5
        kgw.compression = None
        kgw.kernel(nmom_max)

        gw = evGW(self.smf)
        gw.polarizability = "dtda"
        gw.max_cycle = 50
        gw.conv_tol = 1e-8
        gw.damping = 0.5
        gw.compression = None
        gw.kernel(nmom_max)

        self._test_vs_supercell(gw, kgw)

    def test_dtda_vs_supercell_diagonal_w0(self):
        nmom_max = 1

        kgw = evKGW(self.mf)
        kgw.polarizability = "dtda"
        kgw.max_cycle = 200
        kgw.conv_tol = 1e-8
        kgw.diagonal_se = True
        kgw.w0 = True
        kgw.compression = None
        kgw.kernel(nmom_max)

        gw = evGW(self.smf)
        gw.polarizability = "dtda"
        gw.max_cycle = 200
        gw.conv_tol = 1e-8
        gw.diagonal_se = True
        gw.w0 = True
        gw.compression = None
        gw.kernel(nmom_max)

        self._test_vs_supercell(gw, kgw)

    def test_dtda_vs_supercell_g0(self):
        nmom_max = 1

        kgw = evKGW(self.mf)
        kgw.polarizability = "dtda"
        kgw.max_cycle = 5
        kgw.damping = 0.5
        kgw.g0 = True
        kgw.compression = None
        kgw.kernel(nmom_max)

        gw = evGW(self.smf)
        gw.polarizability = "dtda"
        gw.max_cycle = 5
        gw.damping = 0.5
        gw.g0 = True
        gw.compression = None
        gw.kernel(nmom_max)

        self._test_vs_supercell(gw, kgw, full=True, check_convergence=False)


if __name__ == "__main__":
    print("Running tests for evKGW")
    unittest.main()
