from pyscf.agf2.aux_space import GreensFunction
from pyscf.data.nist import HARTREE2EV
from pyscf import gto
from gmtkn import GW100 as systems
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Ellipse
import h5py
import json

ref = "ccsd_t"
ns = (0, 1, 2, 3, 4, 5)

plt.style.use('seaborn-talk')
plt.rc('axes', facecolor='whitesmoke')
plt.rc('figure', facecolor='white')
plt.rc('lines', markeredgecolor='k', markeredgewidth=1.0)


#
# Import data
#

def load_chk(chkfile):
    with h5py.File(chkfile, "r") as f:
        if "ag0w0" in chkfile:
            e = np.array(f["gf"]["energy"])
            c = np.array(f["gf"]["coupling"])
            mu = np.array(f["gf"]["chempot"]).ravel()[0]
        else:
            e = np.array(f["mo_energy"])
            o = np.array(f["mo_occ"])
            c = np.eye(e.size)
            mu = 0.5 * (np.max(e[o > 0]) + np.min(e[o == 0]))
        gf = GreensFunction(e, c, chempot=mu)
        return gf

with open("data/gw100_3c70297.json", "r") as f:
    gw100 = json.load(f)

files_ag0w0 = {
    "diag_ag0w0_hf": "data/diag_ag0w0_%d_df_def2-tzvpp_%s.chk",
    "diag_ag0w0_pbe": "data/diag_ag0w0_%d_df_pbe_def2-tzvpp_%s.chk",
    "diag_ag0w0_b3lyp": "data/diag_ag0w0_%d_df_b3lyp_def2-tzvpp_%s.chk",
    "ag0w0_hf": "data/ag0w0_%d_df_def2-tzvpp_%s.chk",
    "ag0w0_pbe": "data/ag0w0_%d_df_pbe_def2-tzvpp_%s.chk",
    "ag0w0_b3lyp": "data/ag0w0_%d_df_b3lyp_def2-tzvpp_%s.chk",
}
files_g0w0 = {
    "g0w0_ac_hf": "data/g0w0_ac_df_def2-tzvpp_%s.chk",
    "g0w0_ac_pbe": "data/g0w0_ac_df_pbe_def2-tzvpp_%s.chk",
    "g0w0_ac_b3lyp": "data/g0w0_ac_df_b3lyp_def2-tzvpp_%s.chk",
}
labels = {
    "diag_ag0w0_hf": "Diag-A$G_0W_0$@HF",
    "diag_ag0w0_pbe": "Diag-A$G_0W_0$@PBE",
    "diag_ag0w0_b3lyp": "Diag-A$G_0W_0$@B3LYP",
    "ag0w0_hf": "A$G_0W_0$@HF",
    "ag0w0_pbe": "A$G_0W_0$@PBE",
    "ag0w0_b3lyp": "A$G_0W_0$@B3LYP",
    "experiment": "Experiment",
    "ccsd_t": "CCSD(T)",
    "g0w0_ac_hf": "$G_0W_0$@HF",
    "g0w0_ac_pbe": "$G_0W_0$@PBE",
    "g0w0_ac_b3lyp": "$G_0W_0$@B3LYP",
}
names = sorted(list(systems.keys()))
noccs = {}
homos = {
    **{method: {n: [] for n in ns} for method in files_ag0w0.keys()},
    **{method: [] for method in files_g0w0.keys()},
    **{method: [] for method in ("hf", "pbe", "b3lyp", "ccsd_t", "experiment")},
}
lumos = {
    **{method: {n: [] for n in ns} for method in files_ag0w0.keys()},
    **{method: [] for method in files_g0w0.keys()},
    **{method: [] for method in ("hf", "pbe", "b3lyp", "ccsd_t", "experiment")},
}
homo_weights = {
    **{method: {n: [] for n in ns} for method in files_ag0w0.keys()},
    **{method: [] for method in files_g0w0.keys()},
    **{method: [] for method in ("hf", "pbe", "b3lyp", "ccsd_t", "experiment")},
}
lumo_weights = {
    **{method: {n: [] for n in ns} for method in files_ag0w0.keys()},
    **{method: [] for method in files_g0w0.keys()},
    **{method: [] for method in ("hf", "pbe", "b3lyp", "ccsd_t", "experiment")},
}

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
    noccs[key] = nocc

    for method in files_ag0w0.keys():
        for n in ns:
            try:
                gf = load_chk(files_ag0w0[method] % (n, key))
            except:
                homos[method][n].append(np.nan)
                homo_weights[method][n].append(np.nan)
                lumos[method][n].append(np.nan)
                lumo_weights[method][n].append(np.nan)
                print("Error: %s %s %d" % (key, method, n))
                continue

            # homo
            gf_occ = gf.get_occupied()
            arg = np.argmax(gf_occ.coupling[nocc-1]**2)
            homos[method][n].append(-gf_occ.energy[arg] * HARTREE2EV)
            homo_weights[method][n].append(gf_occ.coupling[nocc-1, arg]**2)

            # lumo
            gf_vir = gf.get_virtual()
            arg = np.argmax(gf_vir.coupling[nocc]**2)
            lumos[method][n].append(gf_vir.energy[arg] * HARTREE2EV)
            lumo_weights[method][n].append(gf_vir.coupling[nocc, arg]**2)
            
    for method in files_g0w0.keys():
        try:
            gf = load_chk(files_g0w0[method] % key)
        except Exception as e:
            homos[method].append(np.nan)
            homo_weights[method].append(np.nan)
            lumos[method].append(np.nan)
            lumo_weights[method].append(np.nan)
            print("Error: %s %s" % (key, method))
            continue

        # homo
        gf_occ = gf.get_occupied()
        arg = np.argmax(gf_occ.coupling[nocc-1]**2)
        homos[method].append(-gf_occ.energy[arg] * HARTREE2EV)
        homo_weights[method].append(gf_occ.coupling[nocc-1, arg]**2)

        # lumo
        gf_vir = gf.get_virtual()
        arg = np.argmax(gf_vir.coupling[nocc]**2)
        lumos[method].append(gf_vir.energy[arg] * HARTREE2EV)
        lumo_weights[method].append(gf_vir.coupling[nocc, arg]**2)

    homos["experiment"].append(np.float64(gw100["systems"][key]["experiment"].get("e_ip", np.nan)) * HARTREE2EV)
    lumos["experiment"].append(np.float64(gw100["systems"][key]["experiment"].get("e_ea", np.nan)) * HARTREE2EV)

    homos["ccsd_t"].append(np.float64(gw100["systems"][key]["def2-tzvpp"]["ccsd_t"].get("e_ip", np.nan)) * HARTREE2EV)
    lumos["ccsd_t"].append(np.float64(gw100["systems"][key]["def2-tzvpp"]["ccsd_t"].get("e_ea", np.nan)) * HARTREE2EV)
    
    for mf in ("hf", "pbe", "b3lyp"):
        homos[mf].append(np.float64(gw100["systems"][key]["def2-tzvpp"][mf].get("e_ip", np.nan)) * HARTREE2EV)
        lumos[mf].append(np.float64(gw100["systems"][key]["def2-tzvpp"][mf].get("e_ea", np.nan)) * HARTREE2EV)


#
# Plot scatter
#

fig, ax = plt.subplots(figsize=(6.4, 6.4), facecolor="w")
ax.set_aspect(1.0)
print((" " * 17) + " ".join(["%4d" % n for n in ns]))
for m in sorted(list(files_g0w0.keys())):
    ips = np.array(homos[ref], dtype=np.float64) - np.array(homos[m], dtype=np.float64)
    eas = np.array(lumos[ref], dtype=np.float64) - np.array(lumos[m], dtype=np.float64)
    ip = np.nanmean(np.abs(ips))
    ea = np.nanmean(np.abs(eas))
    ax.plot(ip, ea, "k.")
    ax.annotate(labels[m], (ip, ea), xytext=(ip-0.05, ea-0.05), arrowprops=dict(arrowstyle="-", color="k", lw=2.0, zorder=-1))
    print(("%16s %4d" % (m, np.sum(np.isfinite(np.subtract(ips, eas))))))
outliers = []
for i, xc in enumerate(sorted(list(files_ag0w0.keys()))):
    ips = [np.array(homos[ref], dtype=np.float64) - np.array(homos[xc][n], dtype=np.float64) for n in ns]
    eas = [np.array(lumos[ref], dtype=np.float64) - np.array(lumos[xc][n], dtype=np.float64) for n in ns]
    ip = [np.nanmean(np.abs(x)) for x in ips]
    ea = [np.nanmean(np.abs(x)) for x in eas]
    ax.plot(ip, ea, "C%do-"%i, label=labels[xc], mec="black", ms=12, lw=2)
    # Check outliers
    for name, error_ip, error_ea in zip(names, ips[-1], eas[-1]):
        if error_ip > 1.0 or error_ea > 1.0:
            outliers.append((xc, name, error_ip, error_ea))
    for j, (x, y) in enumerate(zip(ip, ea)):
    #    # Adds an ellipse scaled by the stdv:
    #    e = Ellipse(xy=(x, y), width=np.nanstd(ips)/10, height=np.nanstd(eas)/10)
    #    ax.add_artist(e)
    #    e.set_clip_box(ax.bbox)
    #    e.set_alpha(0.25)
    #    e.set_facecolor("C%d" % i)
        ax.text(x, y, str(ns[j]), ha="center", va="center", c="white", fontsize="small")
    print(("%16s " % xc) + " ".join(["%4d" % np.sum(np.isfinite(np.subtract(x, y))) for x, y in zip(ips, eas)]))
print("Outliers:")
for xc, name, error_ip, error_ea in outliers:
    print(xc, name, error_ip, error_ea)
lim = (min(ax.get_xlim()[0], ax.get_ylim()[0]), max(ax.get_xlim()[1], ax.get_ylim()[1]))
ax.plot(lim, lim, "k-", alpha=0.5, lw=0.5)
ax.set_xlim(lim)
ax.set_ylim(lim)
ax.set_xlabel("Mean absolute error in IP vs. %s (eV)" % labels[ref])
ax.set_ylabel("Mean absolute error in EA vs. %s (eV)" % labels[ref])
ax.grid()
ax.legend(ncol=2)
plt.tight_layout()
plt.savefig("scatter_full.png", dpi=128)


#
# Strip plots vs CCSD(T)
#

violin = True

for diag in [False, True]:
    fig, axs = plt.subplots(nrows=3, ncols=4, figsize=(16, 12), sharey="row", facecolor="w")
    fig.subplots_adjust(wspace=0, hspace=0)

    i = 0
    for method in sorted(files_ag0w0.keys()):
        if not (method.startswith("diag") is diag):
            continue
        ips = [(np.array(homos[ref], dtype=np.float64) - np.array(homos[method][n], dtype=np.float64)) for n in ns]
        eas = [(np.array(lumos[ref], dtype=np.float64) - np.array(lumos[method][n], dtype=np.float64)) for n in ns]
        gaps = [((np.array(homos[ref], dtype=np.float64)+np.array(lumos[ref], dtype=np.float64))-(np.array(homos[method][n], dtype=np.float64)+np.array(lumos[method][n], dtype=np.float64))) for n in ns]
        if violin:
            sns.violinplot(data=ips, ax=axs[0, i], edgecolor=0.5)
            sns.violinplot(data=eas, ax=axs[1, i], edgecolor=0.5)
            sns.violinplot(data=gaps, ax=axs[2, i], edgecolor=0.5)
        else:
            sns.stripplot(data=ips, ax=axs[0, i], edgecolor="black", linewidth=0.5)
            sns.stripplot(data=eas, ax=axs[1, i], edgecolor="black", linewidth=0.5)
            sns.stripplot(data=gaps, ax=axs[2, i], edgecolor="black", linewidth=0.5)
        for ax in axs[:, i]:
            ax.plot([min(ns)-1, max(ns)+1], [0, 0], "k-", lw=0.5)
            ax.set_xlim(min(ns)-0.5, max(ns)+0.5)
        axs[-1, i].set_xlabel(labels[method])
        i += 1

    methods = sorted(files_g0w0.keys())
    ips = [(np.array(homos[ref], dtype=np.float64) - np.array(homos[method], dtype=np.float64)) for method in methods]
    eas = [(np.array(lumos[ref], dtype=np.float64) - np.array(lumos[method], dtype=np.float64)) for method in methods]
    gaps = [((np.array(homos[ref], dtype=np.float64)+np.array(lumos[ref], dtype=np.float64))-(np.array(homos[method], dtype=np.float64)+np.array(lumos[method], dtype=np.float64))) for method in methods]
    if violin:
        sns.violinplot(data=ips, ax=axs[0, -1], edgecolor=0.5)
        sns.violinplot(data=eas, ax=axs[1, -1], edgecolor=0.5)
        sns.violinplot(data=gaps, ax=axs[2, -1], edgecolor=0.5)
    else:
        sns.stripplot(data=ips, ax=axs[0, -1], color="gray", edgecolor="black", linewidth=0.5)
        sns.stripplot(data=eas, ax=axs[1, -1], color="gray", edgecolor="black", linewidth=0.5)
        sns.stripplot(data=gaps, ax=axs[2, -1], color="gray", edgecolor="black", linewidth=0.5)
    for ax in axs[:, -1]:
        ax.plot([-1, len(methods)], [0, 0], "k-", lw=0.5)
        ax.set_xlim(-0.5, len(methods)-0.5)

    axs[0, 0].set_ylabel("MAE in IP vs. %s (eV)" % labels[ref])
    axs[1, 0].set_ylabel("MAE in EA vs. %s (eV)" % labels[ref])
    axs[2, 0].set_ylabel("MAE in Gap vs. %s (eV)" % labels[ref])

    axs[-1, -1].set_xticklabels([labels[method] for method in methods], rotation=15)

    plt.savefig("strip_plot_%svs_ccsdt.png" % ("diag_" if diag else ""), dpi=128)


#
# Strip plots vs GW
#

violin = True

for diag in [False, True]:
    fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(12, 8), sharey="row", facecolor="w")
    fig.subplots_adjust(wspace=0, hspace=0)

    i = 0
    for method in sorted(files_ag0w0.keys()):
        if not (method.startswith("diag") is diag):
            continue
        if diag:
            ref_ = method.replace("diag_ag0w0", "g0w0_ac")
        else:
            ref_ = method.replace("ag0w0", "g0w0_ac")
        ips = [(np.array(homos[ref_], dtype=np.float64) - np.array(homos[method][n], dtype=np.float64)) for n in ns]
        eas = [(np.array(lumos[ref_], dtype=np.float64) - np.array(lumos[method][n], dtype=np.float64)) for n in ns]
        if violin:
            sns.violinplot(data=ips, ax=axs[0, i], edgecolor=0.5)
            sns.violinplot(data=eas, ax=axs[1, i], edgecolor=0.5)
        else:
            sns.stripplot(data=ips, ax=axs[0, i], edgecolor="black", linewidth=0.5)
            sns.stripplot(data=eas, ax=axs[1, i], edgecolor="black", linewidth=0.5)
        for ax in axs[:, i]:
            ax.plot([min(ns)-1, max(ns)+1], [0, 0], "k-", lw=0.5)
            ax.set_xlim(min(ns)-0.5, max(ns)+0.5)
        axs[1, i].set_xlabel(labels[method])
        i += 1

    axs[0, 0].set_ylabel(r"MAE in IP vs. $G_0W_0$ (eV)")
    axs[1, 0].set_ylabel(r"MAE in EA vs. $G_0W_0$ (eV)")

    plt.savefig("strip_plot_%svs_gw.png" % ("diag_" if diag else ""), dpi=128)


#
# Errors vs MF gap size
#

include_ns = ns
plt.figure()
gap_hf = np.array(lumos["hf"]) + np.array(homos["hf"])
gap_pbe = np.array(lumos["pbe"]) + np.array(homos["pbe"])
gap_b3lyp = np.array(lumos["b3lyp"]) + np.array(homos["b3lyp"])
all_gaps = []
all_errors = []
for i, method in enumerate(sorted([x for x in files_ag0w0.keys() if not x.startswith("diag")])):
    gaps = []
    errors = []
    for n in include_ns:
        if method.endswith("hf"):
            errors += list(np.abs((np.array(lumos["g0w0_ac_hf"]) + np.array(homos["g0w0_ac_hf"])) - (np.array(lumos[method][n]) + np.array(homos[method][n]))))
            gaps += list(gap_hf)
        elif method.endswith("pbe"):
            errors += list(np.abs((np.array(lumos["g0w0_ac_pbe"]) + np.array(homos["g0w0_ac_pbe"])) - (np.array(lumos[method][n]) + np.array(homos[method][n]))))
            gaps += list(gap_pbe)
        elif method.endswith("b3lyp"):
            errors += list(np.abs((np.array(lumos["g0w0_ac_b3lyp"]) + np.array(homos["g0w0_ac_b3lyp"])) - (np.array(lumos[method][n]) + np.array(homos[method][n]))))
            gaps += list(gap_b3lyp)
    all_gaps += gaps
    all_errors += errors
    plt.plot(errors, gaps, "C%d."%i, label=labels[method])
plt.xlabel("Error in A$G_0W_0$ gap vs. $GW$ (eV)")
plt.ylabel("Mean-field gap (eV)")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("error_vs_gap.png", dpi=128)


#
# Weird systems
#

tol = 1e-2
diag = True

fig, axs = plt.subplots(nrows=2, ncols=2)
weird_systems = {
    "BN": "10043-11-5",
    "O3": "10028-15-6",
    "MgO": "1309-48-4",
    "BeO": "1304-56-9",
}
keys = sorted(weird_systems.keys())
for i, key in enumerate(keys):
    cas = weird_systems[key]
    #print("%s (%s)" % (name, cas))
    for n in ns:
        #print("niter = %d" % n)
        gf = load_chk(files_ag0w0["%sag0w0_pbe" % ("diag_" if diag else "")] % (n, cas))
        gf_occ = gf.get_occupied()
        #gf_occ.remove_uncoupled(tol=tol)
        es = gf_occ.energy * HARTREE2EV
        wts = [np.linalg.norm(v)**2 for v in gf_occ.coupling.T]
        axs.ravel()[i].scatter([n]*len(es), es, color="k", marker=".", alpha=wts)

axs[0, 0].set_ylabel("Energy (eV)")
axs[1, 0].set_ylabel("Energy (eV)")

axs[1, 0].set_xlabel("Number of iterations")
axs[1, 1].set_xlabel("Number of iterations")

axs[0, 0].set_xticks([])
axs[0, 1].set_xticks([])

for x in (-11.67, -11.42, -11.0):
    axs.ravel()[keys.index("BN")].plot([min(ns)-1, max(ns)+1], [x, x], "k--", lw=0.5)
for x in (-11.95, -11.62, -11.39):
    axs.ravel()[keys.index("O3")].plot([min(ns)-1, max(ns)+1], [x, x], "k--", lw=0.5)
for x in (-9.63, -8.86, -8.62):
    axs.ravel()[keys.index("BeO")].plot([min(ns)-1, max(ns)+1], [x, x], "k--", lw=0.5)
for x in (-7.09, -6.91, -6.66):
    axs.ravel()[keys.index("MgO")].plot([min(ns)-1, max(ns)+1], [x, x], "k--", lw=0.5)

for i in range(2):
    for j in range(2):
        axs[i, j].set_xlim(min(ns)-0.5, max(ns)+0.5)
        axs[i, j].set_ylim(-15.0, -5.0)
        axs[i, j].text(0.8, 0.8, keys[i*2+j], transform=axs[i, j].transAxes, fontsize=12, bbox=dict(facecolor="gray", alpha=0.5))

fig.subplots_adjust(left=0.1, right=0.95, bottom=0.1, top=0.95, wspace=0.15, hspace=0.05)

plt.savefig("weird_systems.png", dpi=128)
