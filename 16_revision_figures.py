"""Revision figures (Figs. 4, 7, 8, 9 of the manuscript) from the saved result files only.

Usage: python 16_revision_figures.py [output_dir]

Every number plotted here is read from LSTM-ODE-BCI/outputs/results.
The ODE profile is recomputed from the saved rates with the same loss as
05_ode_model.py (reproduces the saved loss 0.000702 and the saved
identifiability changes to six digits).
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent
R = BASE / "outputs" / "results"
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else BASE / "outputs" / "figures" / "revision"
OUT.mkdir(parents=True, exist_ok=True)

# Palette (validated reference slots 1-3, light mode) + neutrals
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#9a9893", "#e4e3df"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 9,
    "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5, "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.axisbelow": True, "lines.linewidth": 1.6,
    "savefig.dpi": 300, "pdf.fonttype": 42,
})


def load(name):
    return json.loads((R / name).read_text())


def panel_label(ax, text):
    ax.text(-0.16, 1.04, text, transform=ax.transAxes, fontsize=10,
            fontweight="bold", va="bottom", ha="left", color=INK)


def save(fig, stem):
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.png", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- Figure: subject-level accuracy
tests = load("subject_tests_vs_lstm.json")
subj = {}
for comp, v in tests["comparisons"].items():
    other = comp.split(" vs ")[1]
    for r in v["subject_accuracies"]:
        subj.setdefault(r["subject"], {})["LSTM-Attention"] = r["reference_accuracy"]
        subj[r["subject"]][other] = r["other_accuracy"]
models = ["SVM", "Random Forest", "XGBoost", "EEGNet", "DeepConvNet",
          "CNN-LSTM", "EEG-Transformer", "LSTM-Attention"]
res = {k: v for k, v in load("baseline_results.json").items()}
res.update(load("deep_baselines_results.json"))
res["LSTM-Attention"] = load("lstm_results.json")
subjects = sorted(subj)

fig, ax = plt.subplots(figsize=(7.0, 3.2))
x = np.arange(len(models))
for s in subjects:
    ax.plot(x, [subj[s][m] for m in models], color=MUTED, lw=0.7, alpha=0.55, zorder=1)
    ax.scatter(x, [subj[s][m] for m in models], s=14, color=MUTED, zorder=2,
               edgecolor="white", linewidth=0.4)
for i, m in enumerate(models):
    mean = np.mean([subj[s][m] for s in subjects])
    lo, hi = res[m]["accuracy_ci_low"], res[m]["accuracy_ci_high"]
    col = BLUE if m == "LSTM-Attention" else INK
    ax.errorbar(i + 0.22, mean, yerr=[[mean - lo], [hi - mean]], fmt="o",
                color=col, ms=5, capsize=2.5, lw=1.4, zorder=3)
ax.axvline(2.5, color=GRID, lw=1.0, zorder=0)
ax.text(1.0, 1.005, "Feature-based", ha="center", va="bottom", color=INK2, fontsize=7.5,
        transform=ax.get_xaxis_transform())
ax.text(5.0, 1.005, "Deep end-to-end", ha="center", va="bottom", color=INK2, fontsize=7.5,
        transform=ax.get_xaxis_transform())
ax.set_xticks(x)
ax.set_xticklabels(models, rotation=20, ha="right")
ax.set_ylabel("Subject-level accuracy")
ax.set_ylim(0.6, 1.0)
ax.grid(axis="x", visible=False)
from matplotlib.lines import Line2D
ax.legend(handles=[
    Line2D([], [], color=MUTED, marker="o", ms=4, lw=0.7, label="Individual test subject (n = 8)"),
    Line2D([], [], color=INK, marker="o", ms=5, lw=1.4, label="Mean of subjects, 95% subject-bootstrap CI"),
], loc="lower left", frameon=False)
save(fig, "fig_subject_accuracy")

# ---------------------------------------------------------------- Figure: ODE identifiability
ode = load("ode_results.json")
names = ["k_ap", "k_af", "k_pa", "k_pf", "k_fa", "k_fp"]
k_hat = np.array([ode["parameters"][n] for n in names])
t_obs = np.linspace(0, 30, 20)


def K(k):
    ap, af, pa, pf, fa, fp = k
    return np.array([[-(ap + af), pa, fa], [ap, -(pa + pf), fp], [af, pf, -(fa + fp)]])


def expm(M):
    n = int(max(0, np.ceil(np.log2(max(1e-16, np.abs(M).sum(1).max()))) + 4))
    A = M / 2 ** n
    E = np.eye(3)
    T = np.eye(3)
    for i in range(1, 20):
        T = T @ A / i
        E = E + T
    for _ in range(n):
        E = E @ E
    return E


def mse(k):
    Km = K(k)
    errs = []
    for init, idx in (([1.0, 0, 0], 0), ([0, 0, 1.0], 2)):
        sol = np.array([expm(Km * ti) @ np.array(init) for ti in t_obs])
        errs.append(np.mean((sol[:, idx] - 1.0) ** 2))
    return float(np.mean(errs))


L0 = mse(k_hat)
reg = L0 + 1e-4 * np.sum(k_hat ** 2)
assert abs(reg - ode["fit"]["loss"]) < 1e-7, (reg, ode["fit"]["loss"])
factors = np.logspace(-1, 1, 61)
groups = [
    (["k_af", "k_fa"], r"$k_{AF}$, $k_{FA}$ (Active$\leftrightarrow$Fatigued)", BLUE),
    (["k_ap", "k_fp"], r"$k_{AP}$, $k_{FP}$ (into Passive)", ORANGE),
    (["k_pa", "k_pf"], r"$k_{PA}$, $k_{PF}$ (out of Passive)", AQUA),
]
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8))
ax = axes[0]
for members, label, col in groups:
    i = names.index(members[0])
    rel = []
    for f in factors:
        k = k_hat.copy()
        k[i] = k_hat[i] * f
        rel.append(mse(k) / L0)
    ax.plot(factors, rel, color=col, label=label)
ax.axhspan(0.95, 1.05, color=GRID, alpha=0.9, lw=0, zorder=0)
ax.axvline(1, color=INK2, lw=0.8, ls=":")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_ylim(0.4, 30)
ax.set_yticks([0.5, 1, 2, 5, 10, 20])
ax.set_yticklabels(["0.5", "1", "2", "5", "10", "20"])
ax.minorticks_off()
ax.set_xlabel("Rate relative to fitted value")
ax.set_ylabel("Fit loss relative to optimum")
ax.text(0.11, 1.12, "within 5% of optimum", color=INK2, fontsize=6.6)
ax.legend(loc="upper left", frameon=False, fontsize=6.6)
panel_label(ax, "a")

ax = axes[1]
p_open = 0.5
from_probs = np.array([0.375, 0.25, 0.375])  # initial-state rule at P(open)=0.5
tt = np.linspace(0, 30, 301)
traj = np.array([expm(K(k_hat) * ti) @ from_probs for ti in tt])
for j, (lab, col, ls, lw) in enumerate([("Active", BLUE, "-", 3.2), ("Passive", ORANGE, "-", 1.6),
                                       ("Fatigued", AQUA, "--", 1.6)]):
    ax.plot(tt, traj[:, j], color=col, ls=ls, lw=lw, label=lab)
ax.text(29, 0.465, "Active and Fatigued coincide", ha="right", fontsize=6.6, color=INK2)
ax.set_xlabel("Time (s)")
ax.set_ylabel("Compartment occupancy")
ax.set_ylim(0, 0.6)
ax.legend(loc="center right", frameon=False)
panel_label(ax, "b")
fig.tight_layout(w_pad=2.5)
save(fig, "fig_ode_identifiability")

# ---------------------------------------------------------------- Figure: ODE vs simple alternatives
coup = load("coupling_analysis.json")
temp = load("temporal_baselines.json")["methods"]
fc = load("forecasting_results.json")["metrics"]
fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5))
ax = axes[0]
a = [c["alpha"] for c in coup]
ax.plot(a, [c["accuracy"] for c in coup], "o-", color=BLUE, ms=4, label="Accuracy")
ax.plot(a, [c["f1"] for c in coup], "s-", color=ORANGE, ms=4, label="F1")
ax.set_ylim(0.85, 0.90)
ax.set_xlabel(r"Coupling strength $\alpha$")
ax.set_ylabel("Test score")
ax.legend(frameon=False, loc="lower right")
panel_label(ax, "a")

ax = axes[1]
lab = {"lstm": "LSTM", "moving_average": "Moving average", "majority_vote": "Majority vote",
       "hmm": "HMM", "ode": "One-step ODE"}
offs = {"lstm": (6, 4), "ode": (6, -10), "moving_average": (6, -9), "majority_vote": (6, 4),
        "hmm": (6, 3)}
for key, name in lab.items():
    m = temp[key]
    col = BLUE if key == "ode" else INK
    ax.scatter(m["mean_label_flips_per_second"], m["accuracy_vs_eye_state_label"], s=28,
               color=col, edgecolor="white", linewidth=0.6, zorder=3)
    ax.annotate(name, (m["mean_label_flips_per_second"], m["accuracy_vs_eye_state_label"]),
                textcoords="offset points", xytext=offs[key], fontsize=6.8, color=INK2)
ax.set_xlim(0, 0.26)
ax.set_ylim(0.880, 0.902)
ax.set_xlabel("Label flips per second")
ax.set_ylabel("Accuracy")
panel_label(ax, "b")

ax = axes[2]
hs = sorted(fc, key=lambda h: fc[h]["horizon_seconds"])
sec = [fc[h]["horizon_seconds"] for h in hs]
ax.plot(sec, [fc[h]["ode"]["mae_vs_future_probability"] for h in hs], "o-", color=BLUE, ms=4,
        label="ODE forecast")
ax.plot(sec, [fc[h]["persistence"]["mae_vs_future_probability"] for h in hs], "s--", color=INK2,
        ms=4, label="Persistence")
ax.set_xticks(sec)
ax.set_xlabel("Forecast horizon (s)")
ax.set_ylabel("MAE of future P(eyes closed)")
ax.set_ylim(0.06, 0.09)
ax.legend(frameon=False, loc="upper left")
panel_label(ax, "c")
fig.tight_layout(w_pad=2.0)
save(fig, "fig_ode_alternatives")

# ---------------------------------------------------------------- Figure: ocular audit
oc = load("ocular_spectral.json")
fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.7), gridspec_kw={"width_ratios": [1.05, 1.2, 1.0]})
ax = axes[0]
bars = [("All", oc["full_model_accuracy"], INK2),
        ("Occipital\nzeroed", oc["channel_ablation"]["occipital"]["accuracy"], AQUA),
        ("Frontopolar\nzeroed", oc["channel_ablation"]["frontopolar"]["accuracy"], ORANGE)]
for i, (n, v, c) in enumerate(bars):
    ax.bar(i, v, color=c, width=0.62)
    ax.text(i, v + 0.012, f"{v:.3f}", ha="center", fontsize=7, color=INK)
ax.axhline(0.5, color=INK2, ls="--", lw=0.8)
ax.set_xticks(range(3))
ax.set_xticklabels([b[0] for b in bars], fontsize=6.2)
ax.set_ylim(0, 1.0)
ax.set_ylabel("Test accuracy")
ax.grid(axis="x", visible=False)
panel_label(ax, "a")

ax = axes[1]
grp = ["confident_open", "uncertain", "confident_closed"]
glab = ["Confident\nopen", "Uncertain\n(0.4-0.6)", "Confident\nclosed"]
sb = oc["spectral_by_confidence"]
w = 0.36
for j, (band, col) in enumerate([("theta_relative", ORANGE), ("alpha_relative", BLUE)]):
    means = [sb[g][band]["mean"] for g in grp]
    lo = [sb[g][band]["mean"] - sb[g][band]["ci95"][0] for g in grp]
    hi = [sb[g][band]["ci95"][1] - sb[g][band]["mean"] for g in grp]
    ax.bar(np.arange(3) + (j - 0.5) * w, means, w - 0.03, color=col,
           yerr=[lo, hi], capsize=2, error_kw={"lw": 0.8, "ecolor": INK},
           label="Theta (4-8 Hz)" if j == 0 else "Alpha (8-13 Hz)")
ax.set_xticks(range(3))
ax.set_xticklabels([f"{g}\nn = {sb[k]['alpha_relative']['n']:,}" for g, k in zip(glab, grp)],
                   fontsize=6.6)
ax.set_ylabel("Occipital relative power")
ax.set_ylim(0, 0.42)
ax.grid(axis="x", visible=False)
ax.legend(frameon=False, loc="upper left", ncol=2, fontsize=6.8)
panel_label(ax, "b")

ax = axes[2]
rows = list(csv.DictReader(open(R / "shap_channel_importance.csv")))[:12]
front = {"FP1", "FP2", "FPZ", "AF3", "AF4", "AF7", "AF8", "F7", "F8", "F5", "F6"}
names_c = [r["Channel"] for r in rows][::-1]
vals = [float(r["SHAP_Importance"]) for r in rows][::-1]
cols = [ORANGE if n.upper() in front else MUTED for n in names_c]
ax.barh(range(len(vals)), vals, color=cols, height=0.7)
ax.set_yticks(range(len(vals)))
ax.set_yticklabels(names_c, fontsize=6.8)
ax.set_xlabel("Normalised SHAP importance")
ax.grid(axis="y", visible=False)
ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=ORANGE, label="Anterior / lateral frontal"),
                   plt.Rectangle((0, 0), 1, 1, color=MUTED, label="Other sites")],
          frameon=False, loc="upper center", bbox_to_anchor=(0.45, -0.2), ncol=1, fontsize=6.6)
panel_label(ax, "c")
fig.tight_layout(w_pad=1.8)
save(fig, "fig_ocular_audit")
print("ok", OUT)
