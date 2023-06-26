"""
Tensor Hyper Contraction formulation of the moment GW approach
"""

import numpy as np

from Vayesta.vayesta.rpa.rirpa.NI_eval import NumericalIntegratorBase, NumericalIntegratorGaussianSemiInfinite, \
    NumericalIntegratorClenCurSemiInfinite


class BaseFEval(NumericalIntegratorBase):
    """Class for the numerical integration of F(z) for moment GW. Complete via a double-Laplace transform
    of F(z) integrated over a semi-infinite limit. Utilizes the NI_eval methods from Vayesta rpa.

    Parameters
    ----------
    D : numpy.ndarray
        Array of orbital energy differences.
    npoints : int
        Number of points in the quadrature grid.
    z_point : float
        A point in the z quadrature grid.

    Returns
    -------
    f_calc : tuple
        First element contains the F(z) calculated points for a given (z) value.
        Second element is a float of the error associated with the calculation.
        """
    def __init__(self, D, npoints, z_point, log):
        self.D = D
        self.len_D = self.D.shape[0]
        self.target_rot = np.diag(np.eye(self.len_D))
        self.z_point = z_point
        out_shape = self.target_rot.shape
        diag_shape = (self.len_D,)
        print(f"            Running THC integrand with z={self.z_point}")
        super().__init__(out_shape, diag_shape, npoints, log)

    def eval_contrib(self, freq):
        sinc = np.sin(freq*self.z_point)/self.z_point
        exp = np.exp(-self.D*freq)
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

class FGaussLagEval(
    BaseFEval, NumericalIntegratorGaussianSemiInfinite
):
    pass


class FCCEval(BaseFEval, NumericalIntegratorClenCurSemiInfinite):
    pass


class BaseFIntEval(NumericalIntegratorBase):
    """Class for the numerical integration of F(z) for moment GW. Complete via a double-Laplace transform
    of F(z) integrated over a semi-infinite limit. Utilizes the NI_eval methods from Vayesta rpa.

    Parameters
    ----------
    D : numpy.ndarray
        Array of orbital energy differences.
    npoints : int
        Number of points in the quadrature grid.
    z_point : float
        A point in the z quadrature grid.

    Returns
    -------
    f_calc : tuple
        First element contains the F(z) calculated points for a given (z) value.
        Second element is a float of the error associated with the calculation.
        """
    def __init__(self, D, npoints, z_point, log):
        self.D = D
        self.len_D = self.D.shape[0]
        self.target_rot = np.diag(np.eye(self.len_D))
        self.z_point = z_point
        out_shape = self.target_rot.shape
        diag_shape = (self.len_D,)
        print(f"Int         Running THC integrand with z={self.z_point}")
        super().__init__(out_shape, diag_shape, npoints, log)

    def eval_contrib(self, freq):
        cos = np.cos(self.z_point*freq)/self.D
        exp = np.exp(-self.D*freq)
        res = np.multiply(cos, exp.T)
        return res

    def eval_diag_contrib(self, freq):
        cos = np.cos(self.z_point*freq)/self.D
        exp = np.exp(-self.D * freq)
        return np.multiply(cos, exp.T)

    def eval_diag_deriv_contrib(self, freq):
        deriv_exp = np.exp(-self.D * freq)
        der_cos = self.z_point*np.sin(freq * self.z_point)
        return np.multiply(der_cos, deriv_exp.T)

    def eval_diag_deriv2_contrib(self, freq):
        deriv2_exp = np.multiply(-self.D, np.exp(-self.D * freq))
        der2_cos = np.multiply(self.z_point**2, np.cos(freq * self.z_point))
        return np.multiply(der2_cos, deriv2_exp.T)

    def eval_diag_exact(self):
        return 1 / (self.D ** 2 + self.z_point ** 2)

class FIntGaussLagEval(
    BaseFIntEval, NumericalIntegratorGaussianSemiInfinite
):
    pass


class FIntCCEval(BaseFIntEval, NumericalIntegratorClenCurSemiInfinite):
    pass

class BaseFDEval(NumericalIntegratorBase):
    """Class for the numerical integration of F(z) for moment GW. Complete via a double-Laplace transform
    of F(z) integrated over a semi-infinite limit. Utilizes the NI_eval methods from Vayesta rpa.

    Parameters
    ----------
    D : numpy.ndarray
        Array of orbital energy differences.
    npoints : int
        Number of points in the quadrature grid.
    z_point : float
        A point in the z quadrature grid.

    Returns
    -------
    f_calc : tuple
        First element contains the F(z) calculated points for a given (z) value.
        Second element is a float of the error associated with the calculation.
        """
    def __init__(self, D, npoints, z_point, log):
        self.D = D
        self.len_D = self.D.shape[0]
        self.target_rot = np.diag(np.eye(self.len_D))
        self.z_point = z_point
        self.inv_squ_z = 1/(self.z_point**2)
        out_shape = self.target_rot.shape
        diag_shape = (self.len_D,)
        print(f"D           Running THC integrand with z={self.z_point}")
        super().__init__(out_shape, diag_shape, npoints, log)

    def eval_contrib(self, freq):
        cos = np.cos(self.z_point*freq)
        exp = np.multiply(self.D,np.exp(-self.D*freq))
        res = np.multiply(cos, exp.T)
        return self.inv_squ_z*(1-res)

    def eval_diag_contrib(self, freq):
        cos = np.cos(self.z_point*freq)
        exp = np.multiply(self.D,np.exp(-self.D*freq))
        res = np.multiply(cos, exp.T)
        return self.inv_squ_z*(1-res)

    def eval_diag_deriv_contrib(self, freq):
        der_cos = self.z_point*np.sin(freq * self.z_point)
        der_exp = np.multiply(-self.D**2, np.exp(-self.D * freq))
        res = np.multiply(der_cos, der_exp.T)
        return self.inv_squ_z * res

    def eval_diag_deriv2_contrib(self, freq):
        der2_cos = np.cos(freq * self.z_point)
        der2_exp = np.dot(self.D ** 3, np.exp(-self.D * freq))
        res = np.multiply(der2_cos, der2_exp.T)
        return res

    def eval_diag_exact(self):
        return 1 / (self.D ** 2 + self.z_point ** 2)

class FDGaussLagEval(
    BaseFDEval, NumericalIntegratorGaussianSemiInfinite
):
    pass


class FDCCEval(BaseFDEval, NumericalIntegratorClenCurSemiInfinite):
    pass
