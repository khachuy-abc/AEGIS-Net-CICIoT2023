"""
results_table.py - reads <out>/runs/*.json and writes every table (printed on screen AND saved in <out>/results/):

    summary.md           all tables in one readable file
    summary.csv          mean and std over the seeds, one row per (task, model)
    per_seed.csv         every single run
    per_class_f1.csv     F1 of every class (mean over seeds), raw and calibrated
    per_model/<Model>.md one file per model: its numbers on every task and every class
    finalists.md/.csv    the final comparison: Aegis_net against the 2 strongest ML and the 2 strongest DL models (see finalists.py)

Every number is mean +- standard deviation over the seeds. "Aegis_net vs ..." lines use the SAME seed on both sides.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from .. import settings as S
from ..models.registry import PROPOSED, model_type

NUM = ["f1_macro", "f1_cal", "accuracy", "balanced_accuracy", "mcc", "recall", "fpr", "auc_pr", "fit_seconds", "infer_ms_per_1k"]


def load_runs(out_dir: str) -> pd.DataFrame:
    rows = []
    for f in sorted((Path(out_dir) / "runs").glob("*.json")):
        d = json.load(open(f))
        if "test" not in d:
            continue
        t, c = d["test"], d.get("test_calibrated") or {}
        rows.append(dict(task=d["task"], model=d["model"], seed=d["seed"], f1_macro=t["f1_macro"], f1_cal=c.get("f1_macro", t["f1_macro"]),
                         accuracy=t["accuracy"], balanced_accuracy=t.get("balanced_accuracy"), mcc=t.get("mcc"), recall=t.get("recall"), fpr=t.get("fpr"),
                         auc_pr=t.get("auc_pr"), fit_seconds=d["fit_seconds"], infer_ms_per_1k=d["infer_ms_per_1k"], n_params=d["n_params"],
                         per_class_f1=t.get("per_class_f1"), per_class_f1_cal=c.get("per_class_f1") or t.get("per_class_f1"),
                         per_class_support=t.get("per_class_support"), class_names=d.get("class_names")))
    if not rows:
        raise RuntimeError(f"No finished runs in {out_dir}/runs")
    return pd.DataFrame(rows)


def _pm(m, s, nd=4):
    return "-" if pd.isna(m) else f"{m:.{nd}f} ± {s:.{nd}f}"


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby(["task", "model"]).agg(**{f"{c}_mean": (c, "mean") for c in NUM}, **{f"{c}_std": (c, lambda v: v.std(ddof=0)) for c in NUM},
                                           n_seeds=("seed", "nunique"), n_params=("n_params", "first")).reset_index()


def model_order(models) -> List[str]:
    return [m for m in S.MODEL_ORDER if m in set(models)] + sorted(set(models) - set(S.MODEL_ORDER))


def mean_ranks(agg: pd.DataFrame, col: str = "f1_macro_mean") -> pd.DataFrame:
    """Rank of every model on every task (1 = best) and the mean rank."""
    R = pd.DataFrame({t: g.set_index("model")[col].rank(ascending=False, method="min") for t, g in agg.groupby("task")})
    R["mean_rank"] = R.mean(axis=1)
    return R


def per_class_means(df: pd.DataFrame, col: str = "per_class_f1") -> pd.DataFrame:
    out = []
    for (task, model), g in df.groupby(["task", "model"]):
        vals = [v for v in g[col] if v is not None]
        if not vals:
            continue
        sup = [v for v in g["per_class_support"] if v is not None]
        for i, (c, v) in enumerate(zip(g.class_names.iloc[0], np.mean(vals, axis=0))):
            out.append(dict(task=task, model=model, cls=c, f1=float(v), support=int(sup[0][i]) if sup else 0))
    return pd.DataFrame(out)


def build_tables(out_dir: str, expected_seeds=S.SEEDS) -> str:
    from .finalists import build_finalists
    df = load_runs(out_dir)
    res = Path(out_dir) / "results"
    (res / "per_model").mkdir(parents=True, exist_ok=True)
    agg = aggregate(df)
    df.drop(columns=["per_class_f1", "per_class_f1_cal", "per_class_support", "class_names"]).to_csv(res / "per_seed.csv", index=False)
    agg.to_csv(res / "summary.csv", index=False)
    pc = per_class_means(df).merge(per_class_means(df, "per_class_f1_cal").rename(columns={"f1": "f1_calibrated"}), on=["task", "model", "cls", "support"])
    pc.to_csv(res / "per_class_f1.csv", index=False)

    md: List[str] = ["# Results", "", "F1 macro is the main metric. **F1 raw** = plain argmax decision; **F1 cal.** = after the shared decision-calibration step "
                     "(one log-bias per class fitted on the validation split, identical for every model; the binary task always uses a tuned threshold).", ""]
    bad = agg[agg.n_seeds != len(expected_seeds)]
    if len(bad):
        md += [f"> NOTE: {len(bad)} (task, model) pairs do not have exactly {len(expected_seeds)} seeds: "
               + ", ".join(f"{r.task}/{r.model} ({r.n_seeds})" for r in bad.itertuples()), ""]
    order = model_order(agg.model)
    ranks = mean_ranks(agg)
    for task in [t for t in S.TASKS if t in set(agg.task)]:
        a = agg[agg.task == task].set_index("model").reindex([m for m in order if m in set(agg[agg.task == task].model)])
        rank = ranks[task].astype(int)
        binary = task == S.TASK_BINARY
        md += [f"## Task: {task}", ""]
        head = "| Rank | Type | Model | F1 raw | " + ("" if binary else "F1 cal. | ") + "Accuracy | Balanced acc. | MCC |" + (" Recall | FPR | AUC-PR |" if binary else "") + " Fit time (s) | ms / 1k rows | Params |"
        md += [head, "|" + "---|" * (head.count("|") - 1)]
        for m in a.sort_values("f1_macro_mean", ascending=False).index:
            r = a.loc[m]
            cells = [str(rank[m]), model_type(m), f"**{m}**" if m == PROPOSED else m, _pm(r.f1_macro_mean, r.f1_macro_std)]
            if not binary:
                cells.append(_pm(r.f1_cal_mean, r.f1_cal_std))
            cells += [_pm(r.accuracy_mean, r.accuracy_std), _pm(r.balanced_accuracy_mean, r.balanced_accuracy_std), _pm(r.mcc_mean, r.mcc_std)]
            if binary:
                cells += [_pm(r.recall_mean, r.recall_std), _pm(r.fpr_mean, r.fpr_std), _pm(r.auc_pr_mean, r.auc_pr_std)]
            cells += [f"{r.fit_seconds_mean:.1f}", f"{r.infer_ms_per_1k_mean:.2f}", f"{int(r.n_params):,}"]
            md.append("| " + " | ".join(cells) + " |")
        md += ["", _versus(df[df.task == task], rank), "", _by_type(a), ""]
    md += ["## Overall: rank of every model on every task (F1 raw; 1 = best)", "",
           "| Type | Model | " + " | ".join(t for t in ranks.columns if t != "mean_rank") + " | Mean rank |", "|" + "---|" * (len(ranks.columns) + 2)]
    for m, r in ranks.sort_values("mean_rank").iterrows():
        md.append(f"| {model_type(m)} | {'**' + m + '**' if m == PROPOSED else m} | " + " | ".join(str(int(r[t])) for t in ranks.columns if t != "mean_rank") + f" | {r.mean_rank:.2f} |")
    md.append("")
    (res / "summary.md").write_text("\n".join(md), encoding="utf-8")
    _write_per_model(res / "per_model", agg, pc, ranks)
    fin = build_finalists(out_dir, df, agg, pc)
    return "\n".join(md) + "\n\n" + fin


def _write_per_model(folder: Path, agg: pd.DataFrame, pc: pd.DataFrame, ranks: pd.DataFrame) -> None:
    for m in set(agg.model):
        lines = [f"# {m} ({model_type(m)})", "", "| Task | Rank | F1 raw | F1 cal. | Accuracy | MCC | Fit time (s) | Params |", "|---|---|---|---|---|---|---|---|"]
        for r in agg[agg.model == m].itertuples():
            lines.append(f"| {r.task} | {int(ranks.loc[m, r.task])} | {_pm(r.f1_macro_mean, r.f1_macro_std)} | {_pm(r.f1_cal_mean, r.f1_cal_std)} | "
                         f"{_pm(r.accuracy_mean, r.accuracy_std)} | {_pm(r.mcc_mean, r.mcc_std)} | {r.fit_seconds_mean:.1f} | {int(r.n_params):,} |")
        g = pc[pc.model == m]
        for task, gt in g.groupby("task"):
            lines += ["", f"## F1 per class - {task}", "", "| Class | Test rows | F1 raw | F1 cal. |", "|---|---|---|---|"]
            lines += [f"| {r.cls} | {r.support:,} | {r.f1:.4f} | {r.f1_calibrated:.4f} |" for r in gt.itertuples()]
        (folder / f"{m}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _versus(d: pd.DataFrame, rank: Dict[str, int]) -> str:
    """Aegis_net against the best other model of this task, run by run on identical seeds."""
    if PROPOSED not in set(d.model):
        return ""
    piv = d.pivot_table(index="seed", columns="model", values="f1_macro")
    others = piv.drop(columns=PROPOSED, errors="ignore")
    if others.empty:
        return ""
    best = others.mean().idxmax()
    common = piv[PROPOSED].dropna().index.intersection(others[best].dropna().index)
    diff = piv[PROPOSED][common] - others[best][common]
    beaten = [m for m in others.columns if piv[PROPOSED].mean() > others[m].mean()]
    verdict = "ahead of" if diff.mean() > 0 else "behind"
    return (f"**Aegis_net is rank {rank.get(PROPOSED)} of {len(rank)}.** Best other model: **{best}** ({others[best].mean():.4f}). "
            f"Aegis_net is {verdict} it by {abs(diff.mean()):.4f} F1 macro and wins on {int((diff > 0).sum())} of {len(common)} identical seeds. "
            f"It has a higher mean F1 macro than {len(beaten)} of {len(others.columns)} other models.")


def _by_type(a: pd.DataFrame) -> str:
    """Ranking inside the ML group and inside the DL group (a = per-model means of one task, indexed by model)."""
    lines = []
    for t in ("ML", "DL"):
        members = [m for m in a.index if model_type(m) == t]
        if members:
            o = a.loc[members].sort_values("f1_macro_mean", ascending=False)
            lines.append(f"- **{t} ranking (F1 raw):** " + " > ".join(f"{'**' + m + '**' if m == PROPOSED else m} {o.loc[m, 'f1_macro_mean']:.4f}" for m in o.index))
    return "\n".join(lines)
