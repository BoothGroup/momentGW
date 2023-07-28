import numpy as np
a = 'D'
b = 'Z'
ba = []
ab = []
bab = []

aa = []
baa = []
aab = []
baab = []
baabb = []
aba = []
abab = []

ba.append(b+a)
ab.append(a+b)
aa.append(a+a)

baa += [b + aa for aa in aa]
aab += [aa + b for aa in aa]
aa[0] = a+a+a

aba += [a + ba for ba in ba]
ba = [b + ba for ba in ba]


bab += [b + ab for ab in ab]
ab = [ab + b for ab in ab]
a_power = a+a+a


print('aa',aa)
print('ab',ab)
print('ba',ba)
print('bab',bab)
print('baa',baa)
print('aab',aab)
print('')
print('loop')
for i in range(1):
    new_aa = []
    new_aba = []
    aba_delete = []

    if len(baabb)==0:
        pass
    else:
        baabb = [baabb + b for baabb in baabb]

    new_aba += [a + ba for ba in ba]
    new_aba += [abab + a for abab in abab]
    new_aba += [b + abab for abab in abab]
    baabb += [abab + b for abab in abab]

    if len(aba)==0:
        pass
    else:
        aba_delete += [b + aba for aba in aba]
        abab += [aba+b for aba in aba]
    ba = [b + ba for ba in ba]


    ab = [ab + b for ab in ab]
    ab += [bab +b for bab in bab]
    bab = [b + bab for bab in bab]

    new_aa += [a + baa for baa in baa]
    baa = [b + baa for baa in baa]
    print('new_aa',new_aa)
    new_aa += [aab + a for aab in aab]
    baab = [b + baab for baab in baab]
    baab += [b+aab for aab in aab]

    #if i>0:
    #    baabb += [baab + b for baab in baab]
    #    baab = [b + baab for baab in baab]
    aab = [aab+b for aab in aab]
    baa += [b + aa for aa in aa]
    aab += [aa + b for aa in aa]

    a_power += a
    aa = [a_power] + new_aa
    aba = new_aba

    print('ba', ba)
    print('ab', ab)
    print('bab', bab)
    print('baa', baa)
    print('aab', aab)
    print('baab', baab)
    print('baabb', baabb)
    print('new_aa', aa)
    print('aba', aba)
    print('aba_delete',aba_delete)
    combined = np.asarray(ba + ab + bab + baa + aab + baab + baabb + aa +aba + abab+aba_delete)
    for j in range(combined.shape[0]):
        if len(np.where(combined == combined[j])[0])>1:
            print(combined[j])
    print('All', combined)
    print(len(combined))


