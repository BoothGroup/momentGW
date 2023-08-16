"""
Tests for `pbc/qsgw.py`
"""

import unittest

import numpy as np
import pytest
from pyscf.pbc import gto, dft
from pyscf.pbc.tools import k2gamma
from pyscf.agf2 import mpi_helper

from momentGW import qsGW
from momentGW import qsKGW


class Test_qsKGW(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cell = gto.Cell()
        cell.atom = "He 0 0 0; He 1 1 1"
        cell.basis = "6-31g"
        cell.a = np.eye(3) * 3
        cell.verbose = 0
        cell.precision = 1e-10
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
        if full:
            e1 = np.sort(np.concatenate([gf.energy for gf in kgw.gf]))
            e2 = gw.gf.energy
        else:
            e1 = np.sort(kgw.qp_energy.ravel())
            e2 = gw.qp_energy
        np.testing.assert_allclose(e1, e2, atol=1e-8)

    def test_dtda_vs_supercell(self):
        nmom_max = 3

        kgw = qsKGW(self.mf)
        kgw.polarizability = "dtda"
        kgw.compression = None
        kgw.conv_tol_qp = 1e-10
        kgw.conv_tol = 1e-10
        kgw.eta = 1e-2
        kgw.kernel(nmom_max)

        gw = qsGW(self.smf)
        gw.polarizability = "dtda"
        gw.compression = None
        gw.conv_tol_qp = 1e-10
        gw.conv_tol = 1e-10
        gw.eta = 1e-2
        gw.kernel(nmom_max)

        self._test_vs_supercell(gw, kgw)

    def test_dtda_vs_supercell_srg(self):
        nmom_max = 5

        kgw = qsKGW(self.mf)
        kgw.polarizability = "dtda"
        kgw.srg = 100
        kgw.compression = None
        kgw.conv_tol_qp = 1e-10
        kgw.conv_tol = 1e-10
        kgw.kernel(nmom_max)

        gw = qsGW(self.smf)
        gw.polarizability = "dtda"
        gw.srg = 100
        gw.compression = None
        gw.conv_tol_qp = 1e-10
        gw.conv_tol = 1e-10
        gw.kernel(nmom_max)

        self._test_vs_supercell(gw, kgw)


if __name__ == "__main__":
    print("Running tests for qsKGW")
    unittest.main()
