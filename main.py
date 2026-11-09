"""
main.py - the ONLY command you need.

    python main.py --data_path /path/to/csv_folder_or_file --out outputs

What it does, in order (each step is skipped if its result already exists, so you can re-run after a disconnect):
  1. prepare  : read the CSVs, clean, split train/val/test, scale                              -> <out>/cache
  2. train    : EVERY model x 3 tasks x 5 seeds (neural models: exactly 20 epochs)             -> <out>/runs/*.json
  3. report   : results tables, one file per model, and the FINAL COMPARISON (the 2 strongest ML + the 2 strongest DL models + Aegis_net)
                                                                                               -> <out>/results/

Examples
    python main.py --data_path CICIoT2023/ --out outputs
    python main.py --data_path unsw_nb15.csv --label_column attack_cat --drop_columns id,label --out outputs_unsw
    python main.py --data_path data.csv --models XGBoost LightGBM Aegis_net          # only some models
    python main.py --report_only --out outputs                                       # just rebuild the tables
    python main.py --data_path data.csv --out outputs --prepare_only                  # only build <out>/cache, then stop
Use a different --out for every dataset.
"""
import argparse
import os

from aegis_net import settings as S
from aegis_net.models.registry import MODEL_NAMES
from aegis_net.settings import DataSettings


def parse():
    p = argparse.ArgumentParser(description="Aegis_net benchmark", formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    p.add_argument("--data_path", help="CSV file or folder of CSV files")
    p.add_argument("--out", default="outputs", help="output folder (one per dataset)")
    p.add_argument("--cache_dir", default="", help="where the prepared data goes (default: <out>/cache)")
    p.add_argument("--label_column", default="")
    p.add_argument("--drop_columns", default="", help="comma-separated columns never used as features")
    p.add_argument("--group_labels", default="auto", choices=["auto", "yes", "no"])
    p.add_argument("--max_rows", type=int, default=0, help="read only this many rows (quick tests)")
    p.add_argument("--split_mode", default="block", choices=["block", "row"])
    p.add_argument("--data_seed", type=int, default=S.DATA_SEED, help="seed of the train/validation/test split (change it to repeat an experiment on another split)")
    p.add_argument("--models", nargs="*", default=list(MODEL_NAMES), help=f"default: all of {list(MODEL_NAMES)}")
    p.add_argument("--tasks", nargs="*", default=list(S.TASKS))
    p.add_argument("--fit_rows", type=int, default=S.FIT_ROWS, help="rows every model trains on")
    p.add_argument("--max_test_rows", type=int, default=S.MAX_TEST_ROWS)
    p.add_argument("--epochs", type=int, default=S.EPOCHS, help="official protocol = 20; other values are for smoke tests only")
    p.add_argument("--seeds", type=int, nargs="*", default=list(S.SEEDS), help="official protocol = 5 seeds; other lists are for smoke tests only")
    p.add_argument("--batch_size", type=int, default=S.BATCH_SIZE, help="same for every neural model; lower it on small datasets so the models get enough optimisation steps")
    p.add_argument("--aegis_net_kwargs", default="", help='JSON of Aegis_net constructor overrides (ablations), e.g. \'{"members": 1}\'; baselines are not affected')
    p.add_argument("--force_prepare", action="store_true")
    p.add_argument("--report_only", action="store_true")
    p.add_argument("--prepare_only", action="store_true", help="only read the CSVs and write <out>/cache, then stop")
    return p.parse_args()


def main():
    a = parse()
    S.EPOCHS, S.BATCH_SIZE = a.epochs, a.batch_size
    if a.aegis_net_kwargs:
        import json
        S.AEGIS_NET_KWARGS = json.loads(a.aegis_net_kwargs)
        print(f"[Aegis_net] constructor overrides: {S.AEGIS_NET_KWARGS}")
    official = a.epochs == 20 and tuple(a.seeds) == S.SEEDS
    if not official:
        print(f"[note] epochs={a.epochs}, seeds={a.seeds}: this is NOT the official protocol (20 epochs, seeds {list(S.SEEDS)}); say so when you report these numbers.")
    os.makedirs(a.out, exist_ok=True)
    if not a.report_only:
        from aegis_net.data.dataset import Bundle, Dataset
        from aegis_net.data.prepare import prepare
        from aegis_net.evaluation.runner import run_all
        if not a.data_path:
            raise SystemExit("--data_path is required (unless --report_only)")
        cfg = DataSettings(label_column=a.label_column, drop_columns=tuple(c.strip() for c in a.drop_columns.split(",") if c.strip()),
                           group_labels=a.group_labels, max_rows=a.max_rows, split_mode=a.split_mode, seed=a.data_seed)
        cache = prepare(a.data_path, a.cache_dir or os.path.join(a.out, "cache"), cfg, force=a.force_prepare)
        if a.prepare_only:
            print(f"[data] prepared: {cache}")
            return
        bundle = Bundle(Dataset(cache), fit_rows=a.fit_rows, max_test_rows=a.max_test_rows)
        print(f"[data] {len(bundle.ds.classes)} classes {bundle.ds.classes} | {bundle.ds.n_features} features | fit rows {len(bundle.rows['fit']):,}")
        run_all(bundle, a.out, a.models, a.seeds, a.tasks)
    from aegis_net.evaluation.results_table import build_tables
    print("\n" + build_tables(a.out, a.seeds))
    print(f"\nTables saved in {os.path.join(a.out, 'results')}  (final comparison: finalists.md)")


if __name__ == "__main__":
    main()
