"""Paired per-seed differences of F1 macro between Aegis_net and a competitor.

Reads results/<dataset>/results/per_seed.csv (columns: task, model, seed, f1_macro, f1_cal).
Usage: python tools/paired_seed_diff.py results/ciciot2023/results/per_seed.csv
"""
import csv
import statistics as st
import sys

PAIRS = [("binary", "CatBoost"), ("attack_type", "XGBoost"), ("multiclass", "XGBoost"), ("multiclass", "CatBoost")]


def main(path):
    rows = list(csv.DictReader(open(path)))
    d = {(r["task"], r["model"], int(r["seed"])): r for r in rows}
    seeds = sorted({int(r["seed"]) for r in rows})
    for task, other in PAIRS:
        for key in ("f1_macro", "f1_cal"):
            diffs = [float(d[(task, "Aegis_net", s)][key]) - float(d[(task, other, s)][key]) for s in seeds]
            print(f"{task:12s} vs {other:9s} {key:9s} mean {st.mean(diffs):+.4f} sd {st.stdev(diffs):.4f} "
                  f"wins {sum(x > 0 for x in diffs)}/{len(diffs)}")


if __name__ == "__main__":
    main(sys.argv[1])
