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
        diag_shape = np.diag(np.eye(naux)).shape#self.D.shape
        super().__init__(out_shape, diag_shape, npoints, log)
        self.diagRI = lib.einsum("np,np->n", self.Lia, self.Lia_d)


    def get_offset(self):
        return np.zeros(self.out_shape)

    def eval_contrib(self, freq):
        # This should be real currently, so can safely do this.
        sinc = 2*np.sin(freq*self.z_point)/self.z_point
        lhs = np.multiply(sinc,self.Lia_d)
        rhs = np.multiply(np.exp(-self.D*freq), self.Lia)
        res = np.dot(lhs, rhs.T)
        return res

    def eval_diag_contrib(self, freq):
        sinc = 2*np.sin(freq*self.z_point)/self.z_point
        rhs = np.multiply(np.exp(-self.D * freq), self.Lia_d)
        res = np.dot(self.Lia,rhs.T)
        return np.diag(np.multiply(sinc, res))

    def eval_diag_deriv_contrib(self, freq):
        deriv_exp = np.dot(-self.D,np.exp(-self.D * freq))
        rhs = np.multiply(deriv_exp, self.Lia_d)
        res = np.dot(self.Lia, rhs.T)
        der_sinc = 2 * np.cos(freq * self.z_point)
        return np.diag(np.multiply(der_sinc, res))

    def eval_diag_deriv2_contrib(self, freq):
        deriv2_exp = np.dot(self.D**2, np.exp(-self.D * freq))
        rhs = np.multiply(deriv2_exp, self.Lia_d)
        res = np.dot(self.Lia, rhs.T)
        der2_sinc = -2* self.z_point * np.sin(freq * self.z_point)

        return np.diag(np.multiply(der2_sinc, res))

    def eval_diag_exact(self):
        f = 1.0 / (self.D ** 2 + self.z_point ** 2)
        return np.diag(np.dot(self.Lia * f[None], self.Lia_d.T)) * 4


# class MomzeroOffsetCalcGaussLag(
#     BaseMomzeroQ, NumericalIntegratorGaussianSemiInfinite
# ):
#     pass
#
#
# class MomzeroOffsetCalcCC(BaseMomzeroQ, NumericalIntegratorClenCurSemiInfinite):
#     pass


class BaseMomzeroF(NumericalIntegratorBase):
    def __init__(self, D, npoints, z_point, log):
        self.D = D
        self.len_D = self.D.shape[0]
        self.target_rot = np.diag(np.eye(self.len_D))
        self.z_point = z_point
        out_shape = self.target_rot.shape
        diag_shape = np.diag(np.eye(self.len_D)).shape
        super().__init__(out_shape, diag_shape, npoints, log)

    def eval_contrib(self, freq):
        sinc = np.sin(freq*self.z_point)/self.z_point
        exp = np.multiply(np.exp(-self.D*freq), self.D)
        res = np.dot(sinc, exp.T)
        return res

    def eval_diag_contrib(self, freq):
        sinc = np.sin(freq*self.z_point)/self.z_point
        exp = np.exp(-self.D * freq)#np.multiply(, self.D)
        return np.multiply(sinc, exp.T)

    def eval_diag_deriv_contrib(self, freq):
        deriv_exp = np.dot(-self.D,np.exp(-self.D * freq))
        deriv_exp_D = np.multiply(deriv_exp, self.D)
        der_sinc = np.cos(freq * self.z_point)
        return (np.multiply(der_sinc, deriv_exp.T))

    def eval_diag_deriv2_contrib(self, freq):
        deriv2_exp = np.dot(self.D, np.exp(-self.D * freq))
        deriv2_exp_D = np.multiply(deriv2_exp, self.D)
        der2_sinc = -1 * self.z_point * np.sin(freq * self.z_point)
        return (np.multiply(der2_sinc, deriv2_exp_D.T))

    def eval_diag_exact(self):
        return 1 / (self.D ** 2 + self.z_point ** 2)

class MomzeroOffsetCalcGaussLag(
    BaseMomzeroF, NumericalIntegratorGaussianSemiInfinite
):
    pass


class MomzeroOffsetCalcCC(BaseMomzeroF, NumericalIntegratorClenCurSemiInfinite):
    pass