"""
prepare.py - turns a folder (or a single file) of CSVs into a ready-to-use dataset cache.

    raw CSV files  --read in chunks-->  numeric matrix  --clean-->  --split-->  --scale-->  cache/X.npy, y.npy, role.npy, ...

It does not assume anything about the dataset: number of rows, number of features, number of classes and label names are all
discovered from the files. What you may have to tell it (command line of main.py):
  --label_column NAME     if the label column is not called label / class / attack_cat / ...
  --drop_columns a,b,c    ids, timestamps, or any column that would leak the label
Text columns are one-hot encoded (the N most frequent values + "other"); numeric columns that contain a few text cells keep
the numbers and turn the text cells into NaN (then the median).

Cleaning (all counted in cache/meta.json -> "cleaning"):
  * rows with an empty label are dropped
  * exact duplicate rows are dropped (first one kept)
  * rows with IDENTICAL features but DIFFERENT labels ("collisions") are handled by --collision_policy (default: drop them all)
Splitting: train / validation / test, stratified per class. Consecutive rows are kept together in blocks (default <= 512 rows)
so that near-identical neighbouring flows never end up in both train and test.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import os
import shutil
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..settings import DataSettings
from . import labels as L
from .scaler import Scaler

ROLE_TRAIN, ROLE_VAL, ROLE_TEST = 0, 1, 2
LABEL_NAMES = ("label", "class", "attack_cat", "attack", "attack_type", "category", "target", "y")
CACHE_VERSION = 3


# ------------------------------------------------------------------------------------------------ reading
def discover_files(path: str) -> List[Path]:
    p = Path(path)
    if p.is_file():
        return [p]
    files = sorted(f for f in p.rglob("*") if f.is_file() and f.name.lower().endswith((".csv", ".csv.gz", ".tsv", ".tsv.gz")))
    if not files:
        raise FileNotFoundError(f"No .csv / .csv.gz / .tsv files found under {path!r}")
    return files


def _sep(f: Path) -> str:
    return "\t" if ".tsv" in f.name.lower() else ","


def _read(f: Path, **kw):
    return pd.read_csv(f, sep=_sep(f), low_memory=False, **kw)


def _find(columns, name: str) -> Optional[str]:
    low = {c.strip().lower(): c for c in columns}
    return low.get(name.strip().lower())


def _is_numeric(col: pd.Series) -> bool:
    if pd.api.types.is_numeric_dtype(col):
        return True
    filled = int(col.notna().sum())
    return filled > 0 and pd.to_numeric(col, errors="coerce").notna().sum() >= 0.95 * filled


def _detect_schema(files: List[Path], cfg: DataSettings) -> Dict:
    head = _read(files[0], nrows=50_000)
    head.columns = [str(c).strip() for c in head.columns]
    label = _find(head.columns, cfg.label_column) if cfg.label_column else None
    if label is None:
        if cfg.label_column:
            print(f"[data] label column {cfg.label_column!r} not found; trying common names")
        label = next((c for c in (_find(head.columns, n) for n in LABEL_NAMES) if c), None)
    if label is None:
        raise ValueError(f"Cannot find the label column among {list(head.columns)[:12]}...; pass --label_column NAME")
    drop = {c for c in (_find(head.columns, d) for d in cfg.drop_columns) if c}
    # another label-looking column (e.g. the binary `label` next to `attack_cat` in UNSW-NB15) would leak the answer
    for n in LABEL_NAMES:
        other = _find(head.columns, n)
        if other and other != label and other not in drop:
            print(f"[data] column {other!r} looks like a second label column -> not used as a feature (override with --drop_columns)")
            drop.add(other)
    feats = [c for c in head.columns if c != label and c not in drop]
    numeric = [c for c in feats if _is_numeric(head[c])]
    text = [c for c in feats if c not in numeric]
    return dict(label=label, numeric=numeric, text=text, dropped=sorted(drop))


def _text_vocab(files: List[Path], cols: List[str], cfg: DataSettings) -> Dict[str, List[str]]:
    """Most frequent values of every text column, counted over the whole data (cheap: only those columns are parsed)."""
    counts = {c: Counter() for c in cols}
    seen = 0
    for f in files:
        for chunk in _read(f, usecols=lambda c: str(c).strip() in cols, chunksize=cfg.chunk_rows):
            chunk.columns = [str(c).strip() for c in chunk.columns]
            for c in cols:
                counts[c].update(chunk[c].astype(str).str.strip().tolist())
            seen += len(chunk)
            if cfg.max_rows and seen >= cfg.max_rows:
                break
    return {c: [v for v, _ in counts[c].most_common(cfg.onehot_max_categories)] for c in cols}


def _chunk_to_matrix(chunk: pd.DataFrame, schema: Dict, vocab: Dict[str, List[str]]) -> np.ndarray:
    parts = []
    if schema["numeric"]:
        num = np.array(chunk[schema["numeric"]].apply(pd.to_numeric, errors="coerce"), dtype=np.float32)
        num[~np.isfinite(num)] = np.nan
        parts.append(num)
    for c in schema["text"]:
        v = chunk[c].astype(str).str.strip().to_numpy()
        oh = np.stack([(v == val) for val in vocab[c]], axis=1).astype(np.float32)
        parts.append(np.concatenate([oh, (~oh.any(1)).astype(np.float32)[:, None]], axis=1))
    return np.concatenate(parts, axis=1)


def _feature_names(schema: Dict, vocab: Dict[str, List[str]]) -> List[str]:
    names = list(schema["numeric"])
    for c in schema["text"]:
        names += [f"{c}={v}" for v in vocab[c]] + [f"{c}=<other>"]
    return names


# ------------------------------------------------------------------------------------------------ cleaning
def _dedup(h: np.ndarray, y: np.ndarray, cfg: DataSettings, classes: List[str]) -> Tuple[np.ndarray, Dict]:
    """Boolean keep-mask over rows. h: 64-bit hash of the feature vector, y: class code."""
    n = len(h)
    key = h if cfg.collision_policy != "keep" else h ^ (y.astype(np.uint64) * np.uint64(0x9E3779B97F4A7C15))
    order = np.argsort(key, kind="stable")                    # stable: the first occurrence stays first inside a group
    ks, ys = key[order], y[order].astype(np.int64)
    first = np.r_[True, ks[1:] != ks[:-1]]
    starts = np.flatnonzero(first)
    gid = np.cumsum(first) - 1
    ymin, ymax = np.minimum.reduceat(ys, starts), np.maximum.reduceat(ys, starts)
    conflict_group = ymin != ymax
    keep_sorted = np.ones(n, bool)
    report = dict(rows_in=int(n), exact_duplicates_removed=0, collision_rows_removed=0, collision_groups=int(conflict_group.sum()),
                  collision_policy=cfg.collision_policy, collision_pairs={})
    if conflict_group.any():
        pairs = Counter((classes[a], classes[b]) for a, b in zip(ymin[conflict_group], ymax[conflict_group]))
        report["collision_pairs"] = {f"{a} | {b}": int(c) for (a, b), c in pairs.most_common(15)}
    if cfg.collision_policy == "drop":
        keep_sorted &= ~conflict_group[gid]
    elif cfg.collision_policy == "majority":
        bad = conflict_group[gid]
        if bad.any():
            C = len(classes)
            code = gid[bad].astype(np.int64) * C + ys[bad]
            uc, cnt = np.unique(code, return_counts=True)
            g_of, y_of = uc // C, uc % C
            best_cnt = np.zeros(gid.max() + 1, np.int64)
            np.maximum.at(best_cnt, g_of, cnt)
            n_best = np.zeros_like(best_cnt)
            np.add.at(n_best, g_of, (cnt == best_cnt[g_of]).astype(np.int64))
            win = np.full(gid.max() + 1, -1, np.int64)
            sel = (cnt == best_cnt[g_of]) & (n_best[g_of] == 1)         # a tie has no winner -> the whole group is dropped
            win[g_of[sel]] = y_of[sel]
            keep_sorted &= ~bad | (ys == win[gid])
    if cfg.dedup or cfg.collision_policy != "keep":
        before = int(keep_sorted.sum())
        dup = np.zeros(n, bool)
        # keep only the first surviving row of every group
        idx = np.flatnonzero(keep_sorted)
        g = gid[idx]
        is_first = np.r_[True, g[1:] != g[:-1]]
        dup[idx[~is_first]] = True
        keep_sorted &= ~dup
        report["exact_duplicates_removed"] = before - int(keep_sorted.sum())
    report["collision_rows_removed"] = int(n - report["exact_duplicates_removed"] - keep_sorted.sum())
    keep = np.zeros(n, bool)
    keep[order[keep_sorted]] = True
    report["rows_out"] = int(keep.sum())
    return keep, report


# ------------------------------------------------------------------------------------------------ splitting
def _split_roles(y: np.ndarray, n_classes: int, cfg: DataSettings) -> Tuple[np.ndarray, np.ndarray]:
    """Per-row role (0 train / 1 val / 2 test) and block id. Stratified per class; whole blocks move together."""
    n = len(y)
    block_size = 1 if cfg.split_mode == "row" else int(min(cfg.block_size, max(32, n // 1000)))
    blk = (np.arange(n) // block_size).astype(np.int64)
    rng = np.random.default_rng(cfg.seed)
    role = np.zeros(n, np.uint8)
    for c in range(n_classes):
        rows = np.flatnonzero(y == c)
        if len(rows) < 3:
            continue                                            # too small to split: stays in train
        units, inv = np.unique(blk[rows], return_inverse=True)
        sizes = np.bincount(inv)
        if len(units) < 3:                                      # fewer than 3 blocks: split this class by single rows
            units, inv, sizes = np.arange(len(rows)), np.arange(len(rows)), np.ones(len(rows), np.int64)
        order = rng.permutation(len(units))
        start = np.cumsum(sizes[order]) - sizes[order]
        n_c = len(rows)
        unit_role = np.zeros(len(units), np.uint8)
        unit_role[order] = np.where(start < cfg.test_ratio * n_c, ROLE_TEST, np.where(start < (cfg.test_ratio + cfg.val_ratio) * n_c, ROLE_VAL, ROLE_TRAIN))
        if len(units) >= 3:                                     # guarantee every split contains this class
            unit_role[order[0]], unit_role[order[1]], unit_role[order[-1]] = ROLE_TEST, ROLE_VAL, ROLE_TRAIN
        role[rows] = unit_role[inv]
    return role, (blk // 1).astype(np.int32)


def balanced_subsample(y: np.ndarray, max_rows: int = 0, seed: int = 0) -> np.ndarray:
    """Sorted indices of a subset of y of at most max_rows rows. Every class keeps at most T rows: classes smaller than T are kept
    WHOLE, only big classes are reduced. T is the largest value for which the total stays <= max_rows."""
    y = np.asarray(y)
    n = len(y)
    if n == 0 or max_rows <= 0 or n <= max_rows:
        return np.arange(n)
    counts = np.bincount(y.astype(np.int64))
    counts = counts[counts > 0]
    lo, hi = 1, int(counts.max())
    while lo < hi:
        mid = (lo + hi + 1) // 2
        lo, hi = (mid, hi) if int(np.minimum(counts, mid).sum()) <= max_rows else (lo, mid - 1)
    rng = np.random.default_rng(seed)
    order = np.argsort(y, kind="stable")
    ys = y[order]
    starts = np.flatnonzero(np.r_[True, ys[1:] != ys[:-1]])
    ends = np.r_[starts[1:], n]
    picked = [order[s:e] if e - s <= lo else rng.choice(order[s:e], lo, replace=False) for s, e in zip(starts, ends)]
    return np.sort(np.concatenate(picked))


# ------------------------------------------------------------------------------------------------ main entry
def _signature(files: List[Path], cfg: DataSettings) -> str:
    payload = json.dumps([CACHE_VERSION, [[f.name, f.stat().st_size] for f in files], cfg.__dict__], sort_keys=True, default=str)
    return hashlib.sha1(payload.encode()).hexdigest()[:16]


def prepare(data_path: str, cache_dir: str, cfg: DataSettings, force: bool = False) -> str:
    """Build (or reuse) the cache in `cache_dir`. Returns the cache directory."""
    files = discover_files(data_path)
    sig = _signature(files, cfg)
    out = Path(cache_dir)
    meta_path = out / "meta.json"
    if not force and meta_path.exists() and json.load(open(meta_path)).get("signature") == sig:
        print(f"[data] cache is up to date: {out}")
        return str(out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"[data] reading {len(files)} file(s) from {data_path}")

    schema = _detect_schema(files, cfg)
    print(f"[data] label column: {schema['label']!r} | numeric features: {len(schema['numeric'])} | text features: {len(schema['text'])}"
          + (f" | dropped: {schema['dropped']}" if schema["dropped"] else ""))
    vocab = _text_vocab(files, schema["text"], cfg) if schema["text"] else {}
    feature_names = _feature_names(schema, vocab)
    F = len(feature_names)

    # ---- pass 1: CSV -> raw float32 file on disk + labels + row hashes
    raw_path = out / "_raw.f32"
    labels_raw, hashes, label_vocab, n = [], [], {}, 0
    empty_labels = 0
    with open(raw_path, "wb") as raw:
        for f in files:
            for chunk in _read(f, chunksize=cfg.chunk_rows):
                chunk.columns = [str(c).strip() for c in chunk.columns]
                lab = chunk[schema["label"]].astype(str).str.strip()
                ok = (lab != "") & (lab.str.lower() != "nan")
                empty_labels += int((~ok).sum())
                chunk, lab = chunk[ok.to_numpy()], lab[ok.to_numpy()]
                if cfg.max_rows and n + len(chunk) > cfg.max_rows:
                    chunk, lab = chunk.iloc[: cfg.max_rows - n], lab.iloc[: cfg.max_rows - n]
                if len(chunk) == 0:
                    break
                X = _chunk_to_matrix(chunk, schema, vocab)
                raw.write(np.ascontiguousarray(X).tobytes())
                hashes.append(pd.util.hash_pandas_object(pd.DataFrame(X), index=False).to_numpy(np.uint64))
                labels_raw.append(np.array([label_vocab.setdefault(v, len(label_vocab)) for v in lab.tolist()], dtype=np.int32))
                n += len(chunk)
                print(f"\r[data] rows read: {n:,}", end="", flush=True)
                if cfg.max_rows and n >= cfg.max_rows:
                    break
            if cfg.max_rows and n >= cfg.max_rows:
                break
    print()
    if n == 0:
        raise ValueError("No usable rows: every label cell is empty?")
    y_raw, h = np.concatenate(labels_raw), np.concatenate(hashes)
    raw_names = [None] * len(label_vocab)
    for k, i in label_vocab.items():
        raw_names[i] = k
    modelling_names = L.resolve_class_names(raw_names, cfg.group_labels)
    classes = L.class_order(modelling_names)
    to_code = np.array([classes.index(m) for m in modelling_names], dtype=np.int16)
    y = to_code[y_raw]
    if len(classes) < 2:
        raise ValueError(f"Only one class found ({classes}); nothing to classify.")

    # ---- cleaning
    keep, clean_report = _dedup(h, y, cfg, classes) if (cfg.dedup or cfg.collision_policy != "keep") else (np.ones(n, bool), {})
    clean_report["rows_with_empty_label_skipped"] = empty_labels
    kept_idx = np.flatnonzero(keep)
    yk = y[kept_idx]
    print(f"[data] rows read {n:,} -> kept {len(kept_idx):,} (duplicates removed {clean_report.get('exact_duplicates_removed', 0):,}, "
          f"label collisions removed {clean_report.get('collision_rows_removed', 0):,})")

    # ---- split (on the kept rows, in file order)
    role, blk = _split_roles(yk, len(classes), cfg)

    # ---- scaler fitted on train rows only, class-balanced sample
    raw = np.memmap(raw_path, dtype=np.float32, mode="r", shape=(n, F))
    rng = np.random.default_rng(cfg.seed)
    train_rows = np.flatnonzero(role == ROLE_TRAIN)
    cap = max(1, min(cfg.fit_rows_per_class, 80_000_000 // max(F, 1) // max(len(classes), 1)))
    sample = np.sort(np.concatenate([rng.choice(r, min(len(r), cap), replace=False) for r in
                                     (train_rows[yk[train_rows] == c] for c in range(len(classes))) if len(r)]))
    scaler = Scaler(cfg.clip_quantiles, cfg.skew_log_threshold, cfg.corr_threshold, cfg.zscore_clip).fit(raw[kept_idx[sample]], feature_names)
    F_out = len(scaler.kept)
    print(f"[data] features {F} -> {F_out} after removing constant / highly correlated columns")

    # ---- write the final arrays
    Xout = np.lib.format.open_memmap(out / "X.npy", mode="w+", dtype=np.float32, shape=(len(kept_idx), F_out))
    for s in range(0, len(kept_idx), cfg.chunk_rows):
        Xout[s:s + cfg.chunk_rows] = scaler.transform(raw[kept_idx[s:s + cfg.chunk_rows]])
    Xout.flush()
    del Xout, raw
    os.remove(raw_path)
    np.save(out / "y.npy", yk.astype(np.int16))
    np.save(out / "role.npy", role)
    np.save(out / "block.npy", blk)
    counts = {name: [int(((yk == c) & (role == r)).sum()) for c in range(len(classes))] for name, r in
              (("train", ROLE_TRAIN), ("val", ROLE_VAL), ("test", ROLE_TEST))}
    meta = dict(signature=sig, source=str(data_path), files=[f.name for f in files], label_column=schema["label"], classes=classes,
                raw_labels={k: classes[int(to_code[i])] for k, i in label_vocab.items()}, features=scaler.kept_features,
                rows=int(len(kept_idx)), class_counts=counts, cleaning=clean_report, scaler=scaler.report,
                schema=dict(numeric=len(schema["numeric"]), text=schema["text"], dropped=schema["dropped"]),
                settings={k: (list(v) if isinstance(v, tuple) else v) for k, v in cfg.__dict__.items()})
    json.dump(meta, open(meta_path, "w"), indent=1)
    print(f"[data] classes ({len(classes)}): " + ", ".join(f"{c}={sum(counts[k][i] for k in counts):,}" for i, c in enumerate(classes)))
    return str(out)
