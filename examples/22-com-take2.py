import numpy as np

t = 8
Z = 'Z'
ZZ = "Z'"
D = "D"
b = []
a_only = np.array(['?' for i in range(t)], dtype=object)
a = np.array(["?" for i in range(t)], dtype=object)

a_only[0] = D
b.append(Z)

a_only[1] = D + D
a[1] = [D + b for b in b][0]
b = [Z+ b for b in b]
b += [Z + a_only[0]]

# print(a_only)
# print(a)
# print(b)

print('start')
for i in range(2,t):
    b_temp = [Z + "(" + a[j] + ")" for j in range(1, i)]
    a_temp = D+'('+"+".join([b for b in b]) + ")"
    a = np.roll(a, 1)
    for j in range(2,i+1):
        a[j] = D+a[j]
    a[1] = a_temp
    b = [Z + b for b in b]
    b += b_temp
    b += [Z + a_only[i-1]]
    a_only[i] = D + a_only[i-1]
    q = i
    a_joined = '+'.join([a[s] for s in range(1, i+1)])
    print('+'.join([a_only[q]] + [a_joined] + b))

