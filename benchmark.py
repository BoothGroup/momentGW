import matplotlib.pyplot as plt
import scipy.optimize
import numpy as np

plt.style.use('seaborn-talk')
plt.rc('axes', facecolor='whitesmoke')
plt.rc('figure', facecolor='white')
plt.rc('lines', markeredgecolor='k', markeredgewidth=1.0)

lines = """
OUTPUT  2       mf    2.525
OUTPUT  2    exact    3.619
OUTPUT  2       ac    1.403
OUTPUT  2        0    2.630
OUTPUT  2        5    2.833
OUTPUT  3       mf    0.201
OUTPUT  3    exact   14.378
OUTPUT  3       ac    3.536
OUTPUT  3        0    6.696
OUTPUT  3        5    7.576
OUTPUT  4       mf    0.334
OUTPUT  4    exact   47.369
OUTPUT  4       ac    7.898
OUTPUT  4        0   15.475
OUTPUT  4        5   17.548
OUTPUT  5       mf    0.656
OUTPUT  5    exact  122.495
OUTPUT  5       ac   13.576
OUTPUT  5        0   31.494
OUTPUT  5        5   35.673
OUTPUT  6       mf    1.001
OUTPUT  6    exact  276.196
OUTPUT  6       ac   23.666
OUTPUT  6        0   60.439
OUTPUT  6        5   68.044
OUTPUT  7       mf    1.483
OUTPUT  7    exact  559.164
OUTPUT  7       ac   36.621
OUTPUT  7        0  103.835
OUTPUT  7        5  116.691
OUTPUT  8       mf    2.136
OUTPUT  8       ac   54.145
OUTPUT  8        0  168.859
OUTPUT  8        5  188.728
OUTPUT  9       mf    2.917
OUTPUT  9       ac   84.042
OUTPUT  9        0  265.928
OUTPUT  9        5  295.597
"""
lines = [x.strip() for x in lines.split("\n")]
lines = [x for x in lines if x]

def power(x, prefactor, order):
    return prefactor * x ** order

plt.figure()

f_per_c = 14
f_per_h = 5

for i, (key, name) in enumerate([
        ("mf", "Mean field"),
        ("exact", "Exact GW"),
        ("ac", "AC-GW"),
        #("cd", "CD-GW"),
        ("0", "AGW(0)"),
        ("5", "AGW(5)"),
]):
    nc = []
    time = []
    for line in lines:
        if line.split()[2] == key:
            nc.append(int(line.split()[1]))
            time.append(float(line.split()[3]))

    nao = [(f_per_c * n) + (f_per_h * (2 * n + 4)) for n in nc]

    (prefactor, order), cov = scipy.optimize.curve_fit(power, nao[-3:], time[-3:], p0=(1e-6, 4))
    nao_fit = np.linspace(1.0, max(nao), 256)
    time_fit = power(nao_fit, prefactor, order)

    plt.plot(nao, time, "C%d."%i, label=r"%s" % name)
    plt.plot(nao_fit, time_fit, "C%d-"%i, label=r"$\mathcal{O}[N^{%.3f}]$" % order)

plt.grid()
plt.legend(ncol=2)
handles, labels = plt.gca().get_legend_handles_labels()
plt.legend(handles[::2]+handles[1::2], labels[::2]+labels[1::2], ncol=2)
plt.xlabel("Number of orbitals")
plt.ylabel("Runtime (s)")
#plt.xscale("log")
#plt.yscale("log")
plt.tight_layout()
plt.show()
