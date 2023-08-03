import numpy as np
from scipy.special import binom

import pickle

class THC:
    """
    Compute the self-energy moments using THC integrals in TDA
    Parameters
    ----------
    tda: TDA
        TDA object
    """

    def __init__(
            self,
            tda,
            gw,
    ):
        self.tda = tda
        self.gw = gw
        self.naux = self.tda.naux
        self.nmo = self.tda.nmo
        self.nmom_max = self.tda.nmom_max
        self.total_nmom = self.tda.nmom_max + 1

        self.XiP = self.tda.coll[:self.gw.nocc, :]
        self.XaP = self.tda.coll[self.gw.nocc:, :]
        self.Z = self.tda.cou

        self.ea = self.tda.mo_energy_w[self.tda.mo_occ_w == 0]
        self.ei = self.tda.mo_energy_w[self.tda.mo_occ_w > 0]

    def kernel(self):
        zeta = self.build_THC_zeta()
        moments_occ, moments_vir = self.build_THC_se_moments(zeta)
        return moments_occ, moments_vir

    def build_Z_prime(self):
        Y_i_PQ = np.einsum('iP,iQ->PQ', self.XiP, self.XiP)
        Y_a_PQ = np.einsum('aP,aQ->PQ', self.XaP, self.XaP)
        Z_X_PQ = np.einsum('PQ,PQ->PQ', Y_i_PQ, Y_a_PQ)
        return Z_X_PQ


    def build_THC_zeta(self):
        zeta = np.zeros((self.total_nmom, self.XiP.shape[1], self.XiP.shape[1]))
        ZD_left = np.zeros((self.total_nmom, self.naux, self.naux))
        ZD_only = np.zeros((self.total_nmom, self.naux, self.naux))

        self.Z_prime = self.build_Z_prime()
        self.ZZ = np.einsum('PQ,QR->PR', self.Z, self.Z_prime)

        zeta[0] = self.Z_prime

        YaP = np.einsum('aP,aQ->PQ', self.XaP, self.XaP)
        YiP = np.einsum('aP,aQ->PQ', self.XiP, self.XiP)

        Z_left = np.eye((self.naux))

        for i in range(1,self.total_nmom):
            ZD_left[0] = Z_left
            ZD_left = np.roll(ZD_left, 1, axis=0)

            Z_left = np.einsum('PQ,QR->PR',self.ZZ,Z_left)*2

            Yei_max = np.einsum('i,iP,iQ->PQ', (-1) ** (i) * self.ei ** (i), self.XiP, self.XiP)
            Yea_max = np.einsum('a,aP,aQ->PQ', self.ea ** (i), self.XaP, self.XaP)
            ZD_only[i] = np.einsum('PQ,PQ->PQ', Yea_max, YiP) + np.einsum('PQ,PQ->PQ', Yei_max, YaP)
            ZD_temp = np.zeros((self.naux, self.naux))
            for j in range(1, i):
                Yei = np.einsum('i,iP,iQ->PQ', (-1) ** (j) * self.ei ** (j), self.XiP, self.XiP)
                Yea = np.einsum('a,aP,aQ->PQ', binom(i, j) * self.ea ** (i - j), self.XaP, self.XaP)
                ZD_only[i] += np.einsum('PQ,PQ->PQ', Yea, Yei)
                if j==i-1:
                    Z_left += np.einsum('PQ,QR->PR',self.Z,ZD_only[j])*2
                else:
                    Z_left += np.einsum('PQ,QR,RS->PS',self.Z,ZD_only[i-1-j],ZD_left[i-j])*2
                ZD_temp +=  np.einsum('PQ,QR->PR', ZD_only[j], ZD_left[j])
            zeta[i] = ZD_only[i] + ZD_temp + np.einsum('PQ,QR->PR', self.Z_prime, Z_left)
        return zeta


    def build_THC_se_moments(self,zeta):
        q0, q1 = self.tda.mpi_slice(self.tda.mo_energy_g.size)
        eta = np.zeros((q1 - q0, self.nmom_max + 1, self.nmo, self.nmo))
        for n in range(self.nmom_max + 1):
            zeta_prime = np.einsum('PQ,QR,RS->PS', self.Z, zeta[n], self.Z)
            for x in range(q1 - q0):
                Lp = np.einsum('pP,P->Pp',self.tda.coll,self.tda.coll[x])
                if n==0 and x==1:
                    print(Lp)
                    print(np.einsum('Pp,PQ,Qq->pq', Lp,self.Z,Lp))
                eta[x, n] = np.einsum(f"Pp,Qq,PQ->pq", Lp, Lp, zeta_prime) * 2.0

        moments_occ = np.zeros((self.nmom_max + 1, self.nmo, self.nmo))
        moments_vir = np.zeros((self.nmom_max + 1, self.nmo, self.nmo))
        moms = np.arange(self.total_nmom)
        for n in moms:
            fp = binom(n, moms)
            fh = fp * (-1) ** moms
            if np.any(self.tda.mo_occ_g[q0:q1] > 0):
                eo = np.power.outer(self.tda.mo_energy_g[q0:q1][self.tda.mo_occ_g[q0:q1] > 0], n - moms)
                to = np.einsum(f"t,kt,ktpq->pq", fh, eo, eta[self.tda.mo_occ_g[q0:q1] > 0])
                moments_occ[n] += to
            if np.any(self.tda.mo_occ_g[q0:q1] == 0):
                ev = np.power.outer(self.tda.mo_energy_g[q0:q1][self.tda.mo_occ_g[q0:q1] == 0], n - moms)
                tv = np.einsum(f"t,ct,ctpq->pq", fp, ev, eta[self.tda.mo_occ_g[q0:q1] == 0])
                moments_vir[n] += tv
        moments_occ = 0.5 * (moments_occ + moments_occ.swapaxes(1, 2))
        moments_vir = 0.5 * (moments_vir + moments_vir.swapaxes(1, 2))
        return moments_occ, moments_vir

