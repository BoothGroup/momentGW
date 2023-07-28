import h5py
import numpy as np
import pickle
from scipy.special import comb


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


def Test_Z_X(X_iP,X_aP,Z_PQ):
    Z_X_PQ = Build_Z_X(X_iP, X_aP)
    New = np.einsum('PQ,QR,RS->PS',Z_X_PQ,Z_PQ,Z_X_PQ)
    Lia = np.einsum('iP,aP->Pia',X_iP,X_aP)
    outer = np.einsum('Pia,Qia->PQ',Lia,Lia)
    Old = np.einsum('PQ,QR,RS->PS',outer,Z_PQ,outer)
    print('Test Z_X:',np.allclose(New,Old))


Test_Z_X(X_iP,X_aP,Z_PQ)


def Build_Z_D(X_iP,X_aP,e_a,e_i):
    Y_ei_PQ = np.einsum('i,iP,iQ->PQ', e_i, X_iP, X_iP)
    Y_ea_PQ = np.einsum('a,aP,aQ->PQ', e_a, X_aP, X_aP)
    Y_i_PQ = np.einsum('iP,iQ->PQ',X_iP,X_iP)
    Y_a_PQ = np.einsum('aP,aQ->PQ', X_aP, X_aP)
    Z_ei_PQ = np.einsum('PQ,PQ->PQ', Y_a_PQ, Y_ei_PQ)
    Z_ea_PQ = np.einsum('PQ,PQ->PQ', Y_i_PQ, Y_ea_PQ)
    Z_D_PQ = Z_ea_PQ - Z_ei_PQ
    return Z_D_PQ

def Test_Z_D(X_iP,X_aP,e_a,e_i,D_prop):
    Z_X_PQ = Build_Z_D(X_iP,X_aP,e_a,e_i)
    Lia = np.einsum('iP,aP->Pia',X_iP,X_aP)
    Lia_d = Lia * D_prop[None]
    Old = np.einsum('Pia,Qia->PQ',Lia,Lia_d)
    print('Test Z_D:',np.allclose(Z_X_PQ,Old))

Test_Z_D(X_iP,X_aP,e_a,e_i,D_prop)

def mom_1(X_iP,X_aP,e_a,e_i,Z_PQ):
    Z_X = Build_Z_X(X_iP, X_aP)
    Z_X_prime = np.einsum('PQ,QR,RS->PS',Z_X,Z_PQ,Z_X)
    Z_D = Build_Z_D(X_iP, X_aP, e_a, e_i)
    return Z_X_prime + Z_D

def Test_mom_1(X_iP,X_aP,e_a,e_i,D_prop,Z_PQ):
    New = mom_1(X_iP, X_aP, e_a, e_i, Z_PQ)
    Lia = np.einsum('iP,aP->Pia', X_iP, X_aP)
    outer = np.einsum('Pia,Qia->PQ', Lia, Lia)
    p1 = np.einsum('PQ,QR,RS->PS', outer, Z_PQ, outer)
    Lia_d = Lia * D_prop[None]
    p2 = np.einsum('Pia,Qia->PQ', Lia, Lia_d)
    old = p1 + p2
    print('Test Mom1:', np.allclose(New, old))

Test_mom_1(X_iP,X_aP,e_a,e_i,D_prop,Z_PQ)


def Build_Z_D_n(X_iP,X_aP,e_a,e_i,n):
    Z_D_n = np.zeros((X_iP.shape[1],X_iP.shape[1]))
    if n ==0:
        n = -1
    for i in range(n+1):
        Y_ei_PQ = np.einsum('i,iP,iQ->PQ', (-1)**(i)*e_i**(i), X_iP, X_iP)
        Y_ea_PQ = np.einsum('a,aP,aQ->PQ', comb(n,i)*e_a**(n-i), X_aP, X_aP)
        Z_D_n += np.einsum('PQ,PQ->PQ', Y_ei_PQ, Y_ea_PQ)
    return Z_D_n

def Test_Z_D_1(X_iP, X_aP, e_a, e_i):
    new = Build_Z_D_n(X_iP, X_aP, e_a, e_i, 1)
    old = Build_Z_D(X_iP,X_aP,e_a,e_i)
    print('Test Z^D:', np.allclose(new,old))

Test_Z_D_1(X_iP, X_aP, e_a, e_i)

def Build_Z_D_2(X_iP,X_aP,e_a,e_i):
    Z_D_n = np.zeros((X_iP.shape[1], X_iP.shape[1]))

    Y_ei_PQ = np.einsum('iP,iQ->PQ', X_iP, X_iP)
    Y_ea_PQ = np.einsum('a,aP,aQ->PQ', e_a ** (2), X_aP, X_aP)
    Z_D_n += np.einsum('PQ,PQ->PQ', Y_ei_PQ, Y_ea_PQ)

    Y_ei_PQ = np.einsum('i,iP,iQ->PQ', e_i, X_iP, X_iP)
    Y_ea_PQ = np.einsum('a,aP,aQ->PQ', e_a, X_aP, X_aP)
    Z_D_n -= np.einsum('PQ,PQ->PQ', Y_ei_PQ, Y_ea_PQ)

    Y_ei_PQ = np.einsum('i,iP,iQ->PQ', e_i, X_iP, X_iP)
    Y_ea_PQ = np.einsum('a,aP,aQ->PQ', e_a, X_aP, X_aP)
    Z_D_n -= np.einsum('PQ,PQ->PQ', Y_ei_PQ, Y_ea_PQ)

    Y_ei_PQ = np.einsum('i,iP,iQ->PQ', e_i ** (2), X_iP, X_iP)
    Y_ea_PQ = np.einsum('aP,aQ->PQ', X_aP, X_aP)
    Z_D_n += np.einsum('PQ,PQ->PQ', Y_ei_PQ, Y_ea_PQ)
    return Z_D_n

def Test_Z_D_2(X_iP, X_aP, e_a, e_i):
    new = Build_Z_D_n(X_iP, X_aP, e_a, e_i, 2)
    old = Build_Z_D_2(X_iP,X_aP,e_a,e_i)
    print('Test Z^D^2:', np.allclose(new,old))

Test_Z_D_2(X_iP, X_aP, e_a, e_i)

def Build_inner_mom(X_iP,X_aP,Z_PQ,e_a,e_i,n):
    Z_mom_n = np.zeros((n,X_iP.shape[1], X_iP.shape[1]))
    Z_X = Build_Z_X(X_iP, X_aP)
    Z_Z_PQ = np.einsum('PQ,QR->PR',Z_PQ,Z_X)
    Z_D_n = np.zeros((n,X_iP.shape[1], X_iP.shape[1]))
    Z_D_n[0] = Build_Z_D_n(X_iP, X_aP, e_a, e_i, int(1))
    Z_mom_n[0] = Z_D_n[0] + np.einsum('PQ,QR->PR',Z_X,Z_Z_PQ)
    remain = np.einsum('PQ,QR->PR',Z_X,Z_Z_PQ)
    for j in range(1, n):
        Z_D_n[j] = Build_Z_D_n(X_iP, X_aP, e_a, e_i, int(n+1-j))
        remain = np.einsum('PQ,QR->PR',remain,Z_Z_PQ)
        Z_mom_n[j] = Z_D_n[j] + np.einsum('PQ,QR->PR',Z_mom_n[j-1],Z_Z_PQ) + np.einsum('PQ,QR->PR',Z_Z_PQ.T,Z_mom_n[j-1]) - remain
    return Z_mom_n

# Build_inner_mom(X_iP,X_aP,Z_PQ,e_a,e_i,2)

def Test_inner_mom_2(X_iP,X_aP,Z_PQ,e_a,e_i):
    New = Build_inner_mom(X_iP,X_aP,Z_PQ,e_a,e_i,2)
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

Test_inner_mom_2(X_iP,X_aP,Z_PQ,e_a,e_i)

def Build_Z_D_n_recur(X_iP,X_aP,e_a,e_i,n):
    Z_D_n = np.zeros((n+1,X_iP.shape[1], X_iP.shape[1]))
    if n == 0:
        return np.zeros((X_iP.shape[1], X_iP.shape[1]))
    Y_i_PQ = np.einsum('i,iP,iQ->PQ', X_iP, X_iP)
    Y_a_PQ = np.einsum('a,aP,aQ->PQ', X_aP, X_aP)
    X_i_hang = np.einsum('i,iP->iP', (-1) * e_i, X_iP)
    XY_i_hang = np.einsum('iQ,PQ->iQ',X_i_hang,Y_a_PQ)
    X_a_hang = np.einsum('a,aP->aP', e_a, X_aP)
    XY_a_hang = np.einsum('aQ,PQ->aQ', X_a_hang, Y_i_PQ)
    Y_ei_PQ = np.einsum('iP,iQ->PQ', X_iP, XY_i_hang)
    Y_ea_PQ = np.einsum('aP,aQ->PQ', X_aP, XY_a_hang)
    Z_D_n[1] += np.einsum('PQ,PQ->PQ', Y_ei_PQ, Y_ea_PQ)
    for i in range(2,n + 1):
        Y_ei_PQ = np.einsum('iP,i,iQ->PQ', X_iP, (-1)*e_i, XY_i_hang)
        Y_ea_PQ = np.einsum('aP,a,aQ->PQ', X_aP, XY_a_hang)
        Z_D_n[1] += np.einsum('PQ,PQ->PQ', Y_ei_PQ, Y_ea_PQ)
    return Z_D_n


