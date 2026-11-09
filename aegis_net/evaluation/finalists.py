"""
finalists.py - the final comparison.

After every model has been run, the 2 strongest ML models and the 2 strongest DL models (Aegis_net excluded; strength = mean rank of F1 macro over
the tasks, ties broken by the mean F1) are picked, and compared with Aegis_net in one table per task plus a per-class table that puts the RARE
classes first (they are what Aegis_net is built for). Saved as results/finalists.md and finalists.csv (and finalists_per_class.png if matplotlib exists).
"""
from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from .. import settings as S
from ..models.registry import PROPOSED, model_type


def pick_finalists(agg: pd.DataFrame, per_family: int = S.FINALISTS_PER_FAMILY) -> List[str]:
    from .results_table import mean_ranks
    ranks = mean_ranks(agg)["mean_rank"]
    mean_f1 = agg.groupby("model").f1_macro_mean.mean()
    out = []
    for fam in ("ML", "DL"):
        cand = [m for m in ranks.index if m != PROPOSED and model_type(m) == fam]
        cand.sort(key=lambda m: (ranks[m], -mean_f1[m]))
        out += cand[:per_family]
    return out


def build_finalists(out_dir: str, df: pd.DataFrame, agg: pd.DataFrame, pc: pd.DataFrame) -> str:
    from .results_table import _pm
    res = Path(out_dir) / "results"
    fin = pick_finalists(agg)
    show = fin + ([PROPOSED] if PROPOSED in set(agg.model) else [])
    md = ["# Final comparison", "",
          f"Finalists = the {S.FINALISTS_PER_FAMILY} strongest ML models and the {S.FINALISTS_PER_FAMILY} strongest DL models (by mean rank of F1 macro over the tasks, "
          f"Aegis_net excluded) plus **Aegis_net**.", "", "Picked: " + ", ".join(f"{m} ({model_type(m)})" for m in fin), ""]
    rows = []
    for task in [t for t in S.TASKS if t in set(agg.task)]:
        a = agg[(agg.task == task) & agg.model.isin(show)].set_index("model").loc[[m for m in show if m in set(agg[agg.task == task].model)]]
        md += [f"## {task}", "", "| Model | Type | F1 raw | F1 cal. | Accuracy | Balanced acc. | MCC |", "|---|---|---|---|---|---|---|"]
        for m in a.sort_values("f1_macro_mean", ascending=False).index:
            r = a.loc[m]
            md.append(f"| {'**' + m + '**' if m == PROPOSED else m} | {model_type(m)} | {_pm(r.f1_macro_mean, r.f1_macro_std)} | {_pm(r.f1_cal_mean, r.f1_cal_std)} | "
                      f"{_pm(r.accuracy_mean, r.accuracy_std)} | {_pm(r.balanced_accuracy_mean, r.balanced_accuracy_std)} | {_pm(r.mcc_mean, r.mcc_std)} |")
            rows.append(dict(task=task, model=m, type=model_type(m), f1_raw=r.f1_macro_mean, f1_calibrated=r.f1_cal_mean))
        md.append("")
    g = pc[pc.model.isin(show) & (pc.task == S.TASK_MULTICLASS)]
    if len(g):
        support = g.groupby("cls").support.first().sort_values()                    # rarest class first
        piv = g.pivot(index="cls", columns="model", values="f1").reindex(support.index)[[m for m in show if m in set(g.model)]]
        pivc = g.pivot(index="cls", columns="model", values="f1_calibrated").reindex(support.index)[piv.columns]
        for title, tab in (("F1 per class, raw decision", piv), ("F1 per class, calibrated decision", pivc)):
            md += [f"## {title} (multiclass; rarest class first)", "", "| Class | Test rows | " + " | ".join(f"**{m}**" if m == PROPOSED else m for m in tab.columns) + " |",
                   "|---|---|" + "---|" * len(tab.columns)]
            for c in tab.index:
                best = tab.loc[c].max()
                md.append(f"| {c} | {int(support[c]):,} | " + " | ".join(f"**{v:.3f}**" if v == best else f"{v:.3f}" for v in tab.loc[c]) + " |")
            md.append("")
        _plot(res / "finalists_per_class.png", piv)
    pd.DataFrame(rows).to_csv(res / "finalists.csv", index=False)
    (res / "finalists.md").write_text("\n".join(md), encoding="utf-8")
    return "\n".join(md)


def _plot(path: Path, piv: pd.DataFrame) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(figsize=(1.2 * len(piv) + 4, 4.6))
    w = 0.8 / len(piv.columns)
    for i, m in enumerate(piv.columns):
        ax.bar(np.arange(len(piv)) + i * w, piv[m], w, label=m, edgecolor="black" if m == PROPOSED else None, linewidth=1.5)
    ax.set_xticks(np.arange(len(piv)) + 0.4 - w / 2)
    ax.set_xticklabels(piv.index, rotation=30, ha="right")
    ax.set_ylabel("F1 (raw decision)")
    ax.set_title("Per-class F1 of the finalists (rarest class on the left)")
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=8, ncol=len(piv.columns), loc="upper center", bbox_to_anchor=(0.5, -0.22), frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
