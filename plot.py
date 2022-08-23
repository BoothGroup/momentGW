from pyscf.agf2.aux_space import GreensFunction
from pyscf.data.nist import HARTREE2EV
from pyscf import gto
from gmtkn import GW100 as systems
import numpy as np
import matplotlib.pyplot as plt
import h5py
import json

ref = "ccsd_t"

plt.style.use('seaborn-talk')
plt.rc('axes', facecolor='whitesmoke')
plt.rc('figure', facecolor='white')
plt.rc('lines', markeredgecolor='k', markeredgewidth=1.0)

def load_chk(chkfile):
    with h5py.File(chkfile, "r") as f:
        e = np.array(f["gf"]["energy"])
        c = np.array(f["gf"]["coupling"])
        mu = np.array(f["gf"]["chempot"]).ravel()[0]
        gf = GreensFunction(e, c, chempot=mu)
        return gf

with open("data/gw100_3c70297.json", "r") as f:
    gw100 = json.load(f)

ns = (0, 1, 2, 5)
xcs = {
    "diag": "data/diag_ag0w0_%d_df_def2-tzvpp_%s.chk",
    "hf": "data/ag0w0_%d_df_def2-tzvpp_%s.chk",
    "pbe": "data/ag0w0_%d_df_pbe_def2-tzvpp_%s.chk",
    "b3lyp": "data/ag0w0_%d_df_b3lyp_def2-tzvpp_%s.chk",
}
labels = {
    "diag": "Diag-A$G_0W_0$@HF",
    "hf": "A$G_0W_0$@HF",
    "pbe": "A$G_0W_0$@PBE",
    "b3lyp": "A$G_0W_0$@B3LYP",
    "experiment": "Experiment",
    "ccsd_t": "CCSD(T)",
    "g0w0_hf": "$G_0W_0$@HF",
    "g0w0_pbe": "$G_0W_0$@PBE",
}
ref_methods = ("experiment", "ccsd_t", "g0w0_hf", "hf", "g0w0_pbe", "pbe")
names = sorted(list(systems.keys()))
homos = {xc: {n: [] for n in ns} for xc in xcs.keys()}
lumos = {xc: {n: [] for n in ns} for xc in xcs.keys()}
homo_weights = {xc: {n: [] for n in ns} for xc in xcs.keys()}
lumo_weights = {xc: {n: [] for n in ns} for xc in xcs.keys()}
ref_homos = {ref: [] for ref in ref_methods}
ref_lumos = {ref: [] for ref in ref_methods}

# Get the data:
for key in names:
    mol = gto.M(
            atom=zip(systems[key]['atoms'], systems[key]['coords']),
            charge=systems[key]['charge'],
            spin=systems[key]['spin'],
            basis="def2-tzvpp",
            ecp={'Rb':"def2-tzvpp", 'Ag':"def2-tzvpp", 'I':"def2-tzvpp", 'Cs':"def2-tzvpp", 'Au':"def2-tzvpp", 'Xe':"def2-tzvpp"},
            verbose=0,
    )
    nocc = mol.nelectron // 2

    for xc in xcs.keys():
        for n in ns:
            try:
                gf = load_chk(xcs[xc] % (n, key))
            except:
                homos[xc][n].append(np.nan)
                homo_weights[xc][n].append(np.nan)
                lumos[xc][n].append(np.nan)
                lumo_weights[xc][n].append(np.nan)
                continue

            # homo
            gf_occ = gf.get_occupied()
            arg = np.argmax(gf_occ.coupling[nocc-1]**2)
            homos[xc][n].append(-gf_occ.energy[arg] * HARTREE2EV)
            homo_weights[xc][n].append(gf_occ.coupling[nocc-1, arg]**2)

            # lumo
            gf_vir = gf.get_virtual()
            arg = np.argmax(gf_vir.coupling[nocc]**2)
            lumos[xc][n].append(gf_vir.energy[arg] * HARTREE2EV)
            lumo_weights[xc][n].append(gf_vir.coupling[nocc, arg]**2)

    ref_homos["experiment"].append(np.float64(gw100["systems"][key]["experiment"].get("e_ip", np.nan)) * HARTREE2EV)
    ref_lumos["experiment"].append(np.float64(gw100["systems"][key]["experiment"].get("e_ea", np.nan)) * HARTREE2EV)
    for method in ref_homos.keys():
        if method != "experiment":
            ref_homos[method].append(np.float64(gw100["systems"][key]["def2-tzvpp"][method].get("e_ip", np.nan)) * HARTREE2EV)
            ref_lumos[method].append(np.float64(gw100["systems"][key]["def2-tzvpp"][method].get("e_ea", np.nan)) * HARTREE2EV)


# Plot scatter
if 0:
    fig, ax = plt.subplots(figsize=(6.4, 6.4))
    ax.set_aspect(1.0)
    print((" " * 9) + " ".join(["%4d" % n for n in ns]))
    for m in ("g0w0_hf", "g0w0_pbe"):
        ip = np.nanmean(np.abs(np.array(ref_homos[ref], dtype=np.float64) - np.array(ref_homos[m], dtype=np.float64)))
        ea = np.nanmean(np.abs(np.array(ref_lumos[ref], dtype=np.float64) - np.array(ref_lumos[m], dtype=np.float64)))
        ax.plot(ip, ea, "k.")
        ax.annotate(labels[m], (ip, ea), xycoords="data", xytext=(ip+0.02, ea-0.04), textcoords="data", arrowprops=dict(arrowstyle="-", color="k", lw=1.0))
    for i, xc in enumerate(sorted(list(xcs.keys()))):
        ips = [np.array(ref_homos[ref], dtype=np.float64) - np.array(homos[xc][n], dtype=np.float64) for n in ns]
        eas = [np.array(ref_lumos[ref], dtype=np.float64) - np.array(lumos[xc][n], dtype=np.float64) for n in ns]
        ip = [np.nanmean(np.abs(x)) for x in ips]
        ea = [np.nanmean(np.abs(x)) for x in eas]
        ax.plot(ip, ea, "C%do-"%i, label=labels[xc], mec="black", ms=12, lw=2)
        for j, (x, y) in enumerate(zip(ip, ea)):
            ax.text(x, y, str(ns[j]), ha="center", va="center", c="white", fontsize="small")
        print(("%8s " % xc) + " ".join(["%4d" % np.sum(np.isfinite(np.subtract(x, y))) for x, y in zip(ips, eas)]))
    lim = (min(ax.get_xlim()[0], ax.get_ylim()[0]), max(ax.get_xlim()[1], ax.get_ylim()[1]))
    ax.plot(lim, lim, "k-", alpha=0.5, lw=0.5)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("Mean absolute error in IP vs. %s (eV)" % labels[ref])
    ax.set_ylabel("Mean absolute error in EA vs. %s (eV)" % labels[ref])
    ax.grid()
    ax.legend(ncol=2)
    plt.tight_layout()
    plt.show()


# Plot histograms
if 1:
    fig, axs = plt.subplots(nrows=4, ncols=5, sharex=True, sharey=True, figsize=(6.4, 6.4))
    plt.subplots_adjust(bottom=0.05, top=0.95, left=0.05, right=0.95, wspace=0.0, hspace=0.0)
    ref_gaps = np.array(ref_lumos[ref], dtype=np.float64)-np.array(ref_homos[ref], dtype=np.float64)
    for j, xc in enumerate(sorted(xcs.keys())):
        gaps = [np.array(lumos[xc][n], dtype=np.float64)-np.array(homos[xc][n], dtype=np.float64) for n in ns]
        for i, n in enumerate(ns):
            error = gaps[j] - ref_gaps
            axs[i, j].hist(error, fc="C%d"%i)
    i = 0
    for m in ("hf", "pbe"):
        gaps = np.array(ref_lumos[m], dtype=np.float64)-np.array(ref_homos[m], dtype=np.float64)
        error = gaps - ref_gaps
        axs[i, -1].hist(error, fc="C%d"%(4+i), hatch="/", label=m.upper())
        i += 1
    for m in ("g0w0_hf", "g0w0_pbe"):
        gaps = np.array(ref_lumos[m], dtype=np.float64)-np.array(ref_homos[m], dtype=np.float64)
        error = gaps - ref_gaps
        axs[i, -1].hist(error, fc="C%d"%(4+i-2), hatch="/", label=labels[m])
        i += 1
    for i in range(len(axs)):
        for j in range(len(axs[i])):
            axs[i, j].set_xticks([])
            axs[i, j].set_yticks([])
        axs[-1, i].set_xticks(np.arange(-1.0, 1.1, 1.0))
        if i < 4:
            axs[-1, i].set_xlabel(labels[sorted(xcs.keys())[i]])
        axs[i, 0].set_ylabel(r"$n_\mathrm{iter} = %d$" % ns[i])
    fig.legend(
            bbox_to_anchor=(0.0, 1.02, 1.0, 0.102), loc=3,
            ncol=4, mode="expand", borderaxespad=0.0,
    )
    plt.show()
