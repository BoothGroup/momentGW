"""
Tensor Hyper Contraction formulation of the moment GW approach
"""

import numpy as np

from Vayesta.vayesta.rpa.rirpa.NI_eval import NumericalIntegratorBase, NumericalIntegratorGaussianSemiInfinite, \
    NumericalIntegratorClenCurSemiInfinite
from pyscf import lib


class BaseMomzeroQ(NumericalIntegratorBase):
    def __init__(self, D, Lia, Lia_d, naux, npoints, z_point, log):
        self.D = D
        self.d = D
        self.Lia = Lia
        self.Lia_d = Lia_d
        self.target_rot = np.eye(naux)
        self.z_point = z_point
        out_shape = self.target_rot.shape
        diag_shape = self.D.shape
        #self.npoints(100)
        super().__init__(out_shape, diag_shape, npoints, log)
        self.diagRI = lib.einsum("np,np->p", self.Lia, self.Lia_d)


    def get_offset(self):
        return np.zeros(self.out_shape)

    def eval_contrib(self, freq):
        # This should be real currently, so can safely do this.
        sinc = 2*np.sin(freq*self.z_point)/self.z_point
        lhs = np.multiply(sinc,self.Lia_d)
        rhs = np.multiply(np.exp(-self.D*freq), self.Lia)
        res = np.multiply(lhs, rhs)
        return res

    def eval_diag_contrib(self, freq):
        sinc = 2*np.sin(freq*self.z_point)/self.z_point
        res = np.multiply(np.exp(-self.D * freq), self.diagRI)
        return np.multiply(sinc, res)

    def eval_diag_deriv_contrib(self, freq):
        deriv_exp = np.dot(-self.D,np.exp(-self.D * freq))
        res = np.multiply(deriv_exp, self.diagRI)
        der_sinc = 2 * np.cos(freq * self.z_point)
        return np.multiply(der_sinc, (res))

    def eval_diag_deriv2_contrib(self, freq):
        deriv2_exp = np.dot(self.D**2, np.exp(-self.D * freq))
        res = np.dot(deriv2_exp, self.diagRI)
        der2_sinc = -2* self.z_point * np.sin(freq * self.z_point)
        return np.multiply(der2_sinc, (res))

    def eval_diag_exact(self):
        f = 1.0 / (self.D ** 2 + self.z_point ** 2)
        return np.diag(np.dot(self.Lia * f[None], self.Lia_d.T)) * 4


class MomzeroOffsetCalcGaussLag(
    BaseMomzeroQ, NumericalIntegratorGaussianSemiInfinite
):
    pass

class MomzeroOffsetCalcCC(BaseMomzeroQ, NumericalIntegratorClenCurSemiInfinite):
    pass
