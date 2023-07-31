import h5py
import numpy as np
import pickle
from scipy.special import comb
import cProfile


def LoadData(name_of_pickle:str):
    pickle_file = open(name_of_pickle, 'rb')  # Mode: read + binary
    data_base = pickle.load(pickle_file)
    pickle_file.close()
    return data_base


THC_ERI = h5py.File('/Users/marcusallen/Documents/GitHub/momentGW/thc_eri_LiH/LiH_111/thc_eri_8.h5','r')
coll = np.array(THC_ERI['collocation_matrix']).transpose()[0].transpose()
cou = np.array(THC_ERI['coulomb_matrix'][0]).transpose()[0].T
X_iP = coll[:2,:]
X_aP = coll[2:,:]
Z_PQ = cou

D = LoadData("D")
D_prop = D.reshape(2,9)
e_a = LoadData('e_a')
e_i = LoadData('e_i')

def Build_Z_X(X_iP,X_aP):
    Y_i_PQ = np.einsum('iP,iQ->PQ',X_iP,X_iP)
    Y_a_PQ = np.einsum('aP,aQ->PQ', X_aP, X_aP)
    Z_X_PQ = np.einsum('PQ,PQ->PQ', Y_i_PQ, Y_a_PQ)
    return Z_X_PQ

Z = Z_PQ
Z_prime = Build_Z_X(X_iP,X_aP)
ZZ = np.einsum('PQ,QR->PR',Z,Z_prime)
def Build_Z_D_n(X_iP,X_aP,e_a,e_i,n):
    Z_D_n = np.zeros((X_iP.shape[1],X_iP.shape[1]))
    if n ==0:
        n = -1
    for i in range(n+1):
        Y_ei_PQ = np.einsum('i,iP,iQ->PQ', (-1)**(i)*e_i**(i), X_iP, X_iP)
        Y_ea_PQ = np.einsum('a,aP,aQ->PQ', comb(n,i)*e_a**(n-i), X_aP, X_aP)
        Z_D_n += np.einsum('PQ,PQ->PQ', Y_ei_PQ, Y_ea_PQ)
    return Z_D_n


def Build_moments(Z,ZZ, X_iP,X_aP,e_a,e_i,n):
    naux = Z.shape[0]
    a = np.zeros((n,naux,naux))
    a_only = np.zeros((n,naux,naux))
    moments = np.zeros((n,naux,naux))

    a_only[0] = Build_Z_D_n(X_iP,X_aP,e_a,e_i,1)
    b = ZZ
    moments[0] = a_only[0] + b
    a_temp = np.einsum('PQ,QR->PR',a_only[0],a[1])

    for i in range(1,n):
        a[0] = b
        a = np.roll(a, 1, axis=0)

        b = np.einsum('PQ,QR->PR',ZZ,b)
        a_temp += a_only[i-1]
        b += np.einsum('PQ,QR->PR',Z,a_temp)

        a_only[i] =  Build_Z_D_n(X_iP, X_aP, e_a, e_i, i+1)
        a_temp = np.einsum('mPQ,mQR->PR', a_only[:i], a[1:i+1])
        moments[i] = a_only[i] + a_temp + np.einsum('PQ,QR->PR',Z_prime,b)

    return moments


moments = Build_moments(Z,ZZ, X_iP,X_aP,e_a,e_i,3)
# print('done')

def Test_inner_mom_2(X_iP,X_aP,Z_PQ,e_a,e_i,D_prop):
    New = Build_moments(Z,ZZ, X_iP,X_aP,e_a,e_i,3)
    Lia = np.einsum('iP,aP->Pia', X_iP, X_aP)
    outer = np.einsum('Pia,Qia->PQ', Lia, Lia)
    Lia_d = Lia * D_prop[None]
    p1 = np.einsum('Pia,Qia->PQ',Lia_d,Lia_d)
    p2 = np.einsum('Pia,Qia,QR,RS->PS',Lia,Lia_d,Z_PQ,outer)
    p3 = np.einsum('SR,RP,Pia,Qia->SQ', outer,Z_PQ,Lia_d,Lia)
    inter = np.einsum('PQ,QR->PR',outer,Z_PQ)
    p4 = np.einsum('PQ,QR,RU->PU',inter,inter,outer)
    old = p1+p2+p3+p4
    print('Test 2nd moment:', np.allclose(New[1], old))

def Test_inner_mom_3(X_iP,X_aP,Z_PQ,e_a,e_i,D_prop):
    New = Build_moments(Z,ZZ, X_iP,X_aP,e_a,e_i,3)
    Lia = np.einsum('iP,aP->Pia', X_iP, X_aP)
    outer = np.einsum('Pia,Qia->PQ', Lia, Lia)
    Lia_d = Lia * D_prop[None]
    Lia_d_cont = np.einsum('Pia,Qia->PQ',Lia,Lia_d)
    Lia_d2_cont = np.einsum('Pia,Qia->PQ', Lia_d, Lia_d)
    p1 = np.einsum('Pia,Qia->PQ',Lia_d,Lia_d * D_prop[None])
    p2 = np.einsum('PQ,QR,RS->PS',Lia_d2_cont,Z_PQ,outer)
    p3 = np.einsum('PQ,QR,RS->PS',Lia_d_cont,Z_PQ,Lia_d_cont)
    inter = np.einsum('PQ,QR->PR',Z_PQ,outer)
    p4 = np.einsum('PQ,QR,RU->PU',Lia_d_cont,inter,inter)
    p5 = np.einsum('PQ,QR,RU->PU',outer,Z_PQ,Lia_d2_cont)
    p6 = np.einsum('PQ,QR,RU->PU',inter.T,Lia_d_cont,inter)
    p7 = np.einsum('PQ,QR,RU->PU',inter.T,inter.T,Lia_d_cont)
    p8 = np.einsum('PQ,QR,RU,UT->PT',outer,inter,inter,inter)
    old = p1+p2+p3+p4+p5+p6+p7+p8
    print('Test 3nd moment:', np.allclose(New[2], old))

# Test_inner_mom_2(X_iP,X_aP,Z_PQ,e_a,e_i,D_prop)
# Test_inner_mom_3(X_iP,X_aP,Z_PQ,e_a,e_i,D_prop)


def Build_moments_efficient(Z,ZZ,X_iP,X_aP,e_a,e_i,n):
    naux = Z.shape[0]
    a = np.zeros((n,naux,naux))
    a_only = np.zeros((n,naux,naux))
    moments = np.zeros((n,naux,naux))

    a_only[0] = Build_Z_D_n(X_iP,X_aP,e_a,e_i,1)
    b = ZZ
    moments[0] = a_only[0] + b
    a_temp = np.einsum('PQ,QR->PR',a_only[0],a[1])

    for i in range(1,n):
        a[0] = b
        a = np.roll(a, 1, axis=0)

        b = np.einsum('PQ,QR->PR',ZZ,b)
        a_temp += a_only[i-1]
        b += np.einsum('PQ,QR->PR',Z,a_temp)

        Y_ei_max = np.einsum('i,iP,iQ->PQ', (-1) ** (i+1) * e_i ** (i + 1), X_iP, X_iP)
        Y_a = np.einsum('aP,aQ->PQ', X_aP, X_aP)
        Y_ea_max = np.einsum('a,aP,aQ->PQ', e_a ** (i+1), X_aP, X_aP)
        Y_i = np.einsum('iP,iQ->PQ', X_iP, X_iP)
        a_only[i] = np.einsum('PQ,PQ->PQ', Y_ea_max, Y_i) + np.einsum('PQ,PQ->PQ', Y_ei_max, Y_a)
        a_temp = np.zeros((naux,naux))
        for j in range(1,i+1):
            Y_ei_PQ = np.einsum('i,iP,iQ->PQ', (-1) ** (j) * e_i ** (j), X_iP, X_iP)
            Y_ea_PQ = np.einsum('a,aP,aQ->PQ', comb(i+1, j) * e_a ** (i+1 - j), X_aP, X_aP)
            a_only[i] += np.einsum('PQ,PQ->PQ', Y_ei_PQ, Y_ea_PQ)
            a_temp += np.einsum('PQ,QR->PR', a_only[j-1], a[j])
        moments[i] = a_only[i] + a_temp + np.einsum('PQ,QR->PR',Z_prime,b)

    return moments
n = 2
# moments_old = Build_moments(Z,ZZ, X_iP,X_aP,e_a,e_i,n)
moments_new = Build_moments_efficient(Z,ZZ,X_iP,X_aP,e_a,e_i,n)
print(moments_new)
# print(np.allclose(moments_old[n-1],moments_new[n-1]))
# cProfile.runctx('Build_moments(Z,ZZ, X_iP,X_aP,e_a,e_i,n)',{'Build_moments': Build_moments, 'Z': Z, 'ZZ': ZZ, 'X_iP': X_iP,'X_aP':X_aP, 'e_a':e_a,'e_i':e_i,'n':n}, {})
# cProfile.runctx('Build_moments_efficient(Z,ZZ, X_iP,X_aP,e_a,e_i,n)',{'Build_moments_efficient': Build_moments_efficient, 'Z': Z, 'ZZ': ZZ, 'X_iP': X_iP,'X_aP':X_aP, 'e_a':e_a,'e_i':e_i,'n':n}, {})