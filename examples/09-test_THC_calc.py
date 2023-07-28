import h5py
import numpy as np
import pickle
from scipy.linalg import cholesky
def LoadData(name_of_pickle:str):
    pickle_file = open(name_of_pickle, 'rb')  # Mode: read + binary
    data_base = pickle.load(pickle_file)
    pickle_file.close()
    return data_base

THC_ERI = h5py.File('/Users/marcusallen/Documents/GitHub/momentGW/thc_eri_LiH/LiH_111/thc_eri_8.h5','r')
coll = np.array(THC_ERI['collocation_matrix']).transpose()[0].transpose()
cou = np.array(THC_ERI['coulomb_matrix'][0]).transpose()[0].T
decou = cholesky(cou, lower=True)
Lpq = np.einsum("ip,ap,pq ->qia",coll,coll,decou)
nocc = 2
Lia = Lpq[:,:nocc,nocc:]
Lia = Lia.reshape(Lpq.shape[0],-1)


X_iP = coll[:2,:]
X_aP = coll[2:,:]
Z_PQ = cou
# print(np.allclose(Lia,np.einsum("ip,ap,pq ->qia",X_iP,X_aP,decou).reshape(Lpq.shape[0],-1)))

two_center = np.einsum('iP,aP,PQ,jQ,bQ->iajb',X_iP,X_aP,Z_PQ,X_iP,X_aP)
mult_two = np.einsum('iajb,jbkc->iakc',two_center,two_center)


def N3_two_center(X_iP,X_aP,Z_PQ):
    Z_j_QS = np.einsum('jQ,jS->QS',X_iP,X_iP)
    Z_b_QS = np.einsum('bQ,bS->QS',X_aP,X_aP)
    Z_QS = np.einsum('QS,QS->QS',Z_j_QS,Z_b_QS)
    Z_PS = np.einsum('PQ,QS->PS',Z_PQ,Z_QS)
    Z_PR = np.einsum('PS,SR->PR', Z_PS, Z_PQ)
    return np.einsum('iP,aP,PQ,jQ,bQ->iajb',X_iP,X_aP,Z_PR,X_iP,X_aP)


print('Testing products are the same',np.allclose(mult_two, N3_two_center(X_iP,X_aP,Z_PQ)))




D =LoadData("D")
D_prop = D.reshape(2,9)
print(D_prop.shape)
e_a = LoadData('e_a')
e_i = LoadData('e_i')

def N3_two_center2(X_iP,X_aP,Z_PQ):
    Z_j_QS = np.einsum('jQ,jS->QS',X_iP,X_iP)
    Z_b_QS = np.einsum('bQ,bS->QS',X_aP,X_aP)
    Z_QS = np.einsum('QS,QS->QS',Z_j_QS,Z_b_QS)
    Z_PS = np.einsum('PQ,QS->PS',Z_PQ,Z_QS)
    Z_PR = np.einsum('PS,SR->PR', Z_PS, Z_PQ)

    a = np.einsum('iP,aP,PQ,jQ,bQ->iajb',X_iP,X_aP,Z_PR,X_iP,X_aP)
    b = np.einsum('iP,aP,PQ,jQ,bQ->iajb',X_iP,X_aP,Z_PQ,X_iP,X_aP)
    c = a + b

    Z_PR2 = Z_PR + Z_PQ
    # d = np.einsum('iP,aP,PQ,jQ,bQ->iajb',X_iP2,X_aP2,Z_PR2,X_iP2,X_aP2)
    # print('One less',np.allclose(c,d))

N3_two_center2(X_iP,X_aP,Z_PQ)


mom0 = np.einsum('icjc->ic',mult_two)
mom1 = np.einsum('iajb,jb,jbkc->iakc',two_center,D_prop,two_center)+np.einsum('iajb,jbxy,xykc->iakc',two_center,two_center,two_center)
mom1 = np.einsum('icjc->ic',mom1)
print('mom1.shape',mom1.shape)

Lia_d = Lia * D[None]

def D_prod(e_a,e_i,X_iP,X_aP,Z_PQ):
    X_aP_new = np.multiply(e_a[:, None],X_aP)
    X_aP_new2 = np.einsum('iP,iP->iP',X_aP,X_aP_new)#X_aP_new + X_aP
    print('X_aP_new',X_aP_new.shape)
    X_iP_new = np.multiply(-e_i[:, None], X_iP)
    X_iP_new2 = np.einsum('iP,iP->iP',X_iP,X_iP_new)#X_iP +X_iP_new
    print(np.allclose(X_iP_new2, np.einsum('iP,iP->iP',X_iP,X_iP_new)))
    print('X_iP_new',X_iP_new.shape)
    Lia_n3 = np.einsum('iP,aP,PQ,jQ,bQ->iajb', X_iP_new2, X_aP_new2, Z_PQ, X_iP, X_aP)
    X_ia1 = np.einsum('iP,aP->Pia', X_iP_new, X_aP)
    X_ia2 = np.einsum('iP,aP->Pia', X_iP, X_aP_new)
    X_ia3 = X_ia1 + X_ia2
    Lia_n3 = np.einsum('Pia,PQ,jQ,bQ->iajb', X_ia3, Z_PQ, X_iP, X_aP)
    Lia_1 = np.einsum('iP,aP,PQ,jQ,bQ->iajb', X_iP_new, X_aP, Z_PQ, X_iP, X_aP)
    Lia_2 = np.einsum('iP,aP,PQ,jQ,bQ->iajb', X_iP, X_aP_new, Z_PQ, X_iP, X_aP)
    print(np.allclose(Lia_n3,Lia_1+Lia_2))
    Lia_n = Lia_1+Lia_2
    print(Lia_n.shape)
    Lia_n = Lia_n.reshape(Lia_n.shape[0],Lia_n.shape[1],-1)
    print(Lia_n.shape)
    Lia_n = Lia_n.reshape(-1, Lia_n.shape[2])
    print(Lia_n.shape)
    return Lia_n

# print(X_aP)
# print(e_a)
# print(np.multiply(e_a[:, None],X_aP)/X_aP)
D_two = np.dot(Lia_d.T, Lia)
print(np.where(D_two-D_prod(e_a,e_i,X_iP,X_aP,Z_PQ)>1e-05))

print('D product', np.allclose(D_prod(e_a,e_i,X_iP,X_aP,Z_PQ),D_two))

def decomp(X_iP,X_aP):
    Lia = np.einsum('iP,aP',X_iP)

def SVD(X_iP,X_aP,Z_PQ):
    S1, V1, D1 = np.linalg.svd(X_iP, full_matrices=False)
    S2, V2, D2 = np.linalg.svd(X_aP, full_matrices=False)
    print(np.diag(V1).shape)
    print(np.allclose(np.einsum('ab,bc,cP->aP', S1, np.diag(V1), D1),X_iP))
    recon = np.einsum('ab,bc,cP,ij,jk,kP->Pai', S1, np.diag(V1), D1, S2, np.diag(V2), D2)
    print(np.allclose(recon, np.einsum('aP,iP->Pia',X_aP,X_iP)))


SVD(X_iP,X_aP,Z_PQ)

print('')
print('start')

print(D_prop.shape)

def attempt(X_iP,X_aP,e_a,e_i,Z_PQ,D_prop):
    X_e_iP = np.einsum('i,iP->iP', e_i, X_iP)
    X_e_aP = np.einsum('i,iP->iP', e_a, X_aP)
    X_e_iP_c = np.einsum('iP,iQ->PQ', X_iP, X_e_iP)
    X_e_aP_c = np.einsum('aP,aQ->PQ', X_aP, X_e_aP)
    X_iP_c = np.einsum('iP,iQ->PQ', X_iP, X_iP)
    X_aP_c = np.einsum('aP,aQ->PQ', X_aP, X_aP)
    Z_i_PQ = np.einsum('PQ,PQ->PQ', X_e_iP_c, X_aP_c)
    Z_a_PQ = np.einsum('PQ,PQ->PQ', X_e_aP_c, X_iP_c)
    Z_PQ_new =Z_a_PQ - Z_i_PQ
    print(np.mean(Z_PQ_new))
    # X_D_aP = np.einsum('ia,aP->iP',D_prop, X_aP)
    # Z_iei_PQ = np.einsum('Pa,aQ->PQ', X_D_Pi, X_aP)
    # Z_aea_PQ =  np.einsum('Pi,iQ->PQ', X_D_Pa, X_iP)
    # Z_ea = np.einsum('PQ,PQ->PQ',Z_iei_PQ,Z_aea_PQ)
    test1 = np.einsum('iP,aP->Pia',X_iP, X_aP)
    test2 = test1* D_prop[None]
    #rebuild =np.einsum('aP,iP->Pia', X_D_iP,X_D_aP)
    # print(test2-rebuild)
    testing = np.einsum('Pia,Qia->PQ', test1,test2)
    print(np.allclose(Z_PQ_new,testing))
    # print(np.allclose(Z_iei_PQ,Z_aea_PQ))
    # print(Z_iei_PQ)
    # print(Z_iei_PQ.shape)
    # print(X_D_aP)
    print('')
    # print(Z_ea)
    print('break')
    # print(X_iP.T)
    # print(X_iP.shape)
    # print(X_aP)
    # print(test1)
    # print(test2)
    # print((testing))
    # midstep = np.einsum('PQ,QR->PR', Z_PQ,Z_spec)
    # midstep2 = np.einsum('PQ,QR->PR', midstep,Z_PQ)
    # print('here')
    # final = np.einsum('iP,aP,PS,jS,bS->iajb',X_iP,X_aP,midstep2,X_iP,X_aP)
    # print(final.shape)
    # return final

attempt(X_iP,X_aP,e_a,e_i,Z_PQ,D_prop)
# Testing = np.einsum()


def Z_D(X_iP,X_aP,e_a,e_i,):
    X_e_iP = np.einsum('i,iP->iP', e_i, X_iP)
    X_e_aP = np.einsum('i,iP->iP', e_a, X_aP)
    X_e_iP_c = np.einsum('iP,iQ->PQ', X_iP, X_e_iP)
    X_e_aP_c = np.einsum('aP,aQ->PQ', X_aP, X_e_aP)
    X_iP_c = np.einsum('iP,iQ->PQ', X_iP, X_iP)
    X_aP_c = np.einsum('aP,aQ->PQ', X_aP, X_aP)
    Z_i_PQ = np.einsum('PQ,PQ->PQ', X_e_iP_c, X_aP_c)
    Z_a_PQ = np.einsum('PQ,PQ->PQ', X_e_aP_c, X_iP_c)
    Z_PQ_new =Z_a_PQ - Z_i_PQ
    print(np.mean(Z_PQ_new))