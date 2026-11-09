import sys, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
out = sys.argv[1]
INK, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#ffffff"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": MUTED,
                     "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": MUTED})
COL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]

# ---- Figure 1: per-class F1 (raw), rarest class first
classes = ["BruteForce\n(1,875)", "Web-based\n(3,538)", "Spoofing\n(65,161)", "Recon\n(97,971)",
           "Benign\n(156,929)", "Mirai\n(353,162)", "DoS\n(528,626)", "DDoS\n(1,716,959)"]
data = {
 "AEGIS-Net": [0.369, 0.348, 0.897, 0.780, 0.859, 0.997, 0.672, 0.886],
 "XGBoost":   [0.375, 0.320, 0.897, 0.778, 0.856, 0.997, 0.678, 0.883],
 "CatBoost":  [0.379, 0.307, 0.894, 0.772, 0.852, 0.997, 0.677, 0.884],
 "MLP-Res":   [0.378, 0.268, 0.849, 0.763, 0.837, 0.997, 0.658, 0.878],
 "CNN1D":     [0.355, 0.241, 0.846, 0.757, 0.831, 0.997, 0.660, 0.876],
}
fig, ax = plt.subplots(figsize=(7.2, 3.6), dpi=200, facecolor=SURF)
n = len(data); w = 0.16; x = np.arange(len(classes))
for i, (k, v) in enumerate(data.items()):
    ax.bar(x + (i - (n-1)/2) * w, v, w, label=k, color=COL[i], edgecolor=SURF, linewidth=0.8, zorder=3)
ax.set_xticks(x); ax.set_xticklabels(classes, fontsize=7.5)
ax.set_ylim(0, 1.05); ax.set_ylabel("F1 score (raw decision)")
ax.yaxis.grid(True, color=GRID, linewidth=0.7, zorder=0); ax.set_axisbelow(True)
for s in ("top", "right"): ax.spines[s].set_visible(False)
ax.legend(ncol=5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.13), fontsize=8, handlelength=1.2)
ax.set_xlabel("Attack family (test rows), rarest first", color=MUTED, fontsize=8)
fig.tight_layout(); fig.savefig(f"{out}/fig_perclass.pdf"); fig.savefig(f"{out}/fig_perclass.png"); plt.close(fig)

# ---- Figure 2: ablation, signed change vs full model
var = ["Full model", "No shrinkage", "No bins\n(levels = 0)", "1 member\n(no ensemble)"]
d_all  = [0.0, -0.0139, -0.0218, 0.0009]
d_rare = [0.0, -0.0183, -0.0351, 0.0035]
fig, ax = plt.subplots(figsize=(6.2, 3.3), dpi=200, facecolor=SURF)
x = np.arange(len(var)); w = 0.3
for off, d, c, lab in ((-w/2-0.01, d_all, COL[0], "Macro F1, all 8 classes"),
                       (w/2+0.01, d_rare, COL[1], "Mean F1, 3 rarest classes")):
    b = ax.bar(x + off, d, w, color=c, edgecolor=SURF, linewidth=0.8, label=lab, zorder=3)
    for xi, v in zip(x + off, d):
        ax.text(xi, v + (0.0012 if v >= 0 else -0.0012), f"{v:+.4f}" if v else "0", ha="center",
                va="bottom" if v >= 0 else "top", fontsize=7.5, color=INK)
ax.axhline(0, color=MUTED, linewidth=0.9, zorder=4)
ax.set_xticks(x); ax.set_xticklabels(var, fontsize=8)
ax.set_ylim(-0.043, 0.010); ax.set_ylabel("Change in F1 vs. full model")
ax.yaxis.grid(True, color=GRID, linewidth=0.7, zorder=0); ax.set_axisbelow(True)
for s in ("top", "right", "bottom"): ax.spines[s].set_visible(False)
ax.tick_params(axis="x", length=0)
ax.legend(ncol=2, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.14), fontsize=8, handlelength=1.2)
fig.tight_layout(); fig.savefig(f"{out}/fig_ablation.pdf"); fig.savefig(f"{out}/fig_ablation.png"); plt.close(fig)
