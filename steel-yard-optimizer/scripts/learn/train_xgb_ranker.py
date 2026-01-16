from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

FEATURES = [
    "thickness_mm","width_mm","length_mm","weight_ton","level","blocked_cnt","is_top",
    "stack_height","stack_count","area_hash","stack_hash",
    "step","demand_pos","remaining_pos","remaining_cnt",
]

def load_groups(p: Path) -> List[int]:
    txt = p.read_text(encoding="utf-8").strip()
    if not txt:
        return []
    return [int(x) for x in txt.splitlines() if x.strip()]

def split_by_groups(df: pd.DataFrame, groups: List[int], test_size: float, seed: int) -> Tuple[pd.DataFrame, List[int], pd.DataFrame, List[int]]:
    idx = np.arange(len(groups))
    tr_idx, va_idx = train_test_split(idx, test_size=test_size, random_state=seed, shuffle=True)
    tr_set = set(tr_idx.tolist())
    va_set = set(va_idx.tolist())

    start = 0
    tr_chunks = []
    va_chunks = []
    g_tr: List[int] = []
    g_va: List[int] = []
    for gi, gsz in enumerate(groups):
        end = start + gsz
        if gi in tr_set:
            tr_chunks.append(df.iloc[start:end])
            g_tr.append(gsz)
        elif gi in va_set:
            va_chunks.append(df.iloc[start:end])
            g_va.append(gsz)
        start = end

    df_tr = pd.concat(tr_chunks, axis=0) if tr_chunks else df.iloc[0:0].copy()
    df_va = pd.concat(va_chunks, axis=0) if va_chunks else df.iloc[0:0].copy()
    return df_tr, g_tr, df_va, g_va

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/datasets/ltr.parquet")
    ap.add_argument("--groups", default="data/datasets/ltr.groups.txt")
    ap.add_argument("--out", default="models/xgb_ranker.json")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--ndcg-k", type=int, default=10)
    ap.add_argument("--rounds", type=int, default=8000)
    ap.add_argument("--early-stop", type=int, default=300)
    ap.add_argument("--max-bin", type=int, default=256)
    ap.add_argument("--max-leaves", type=int, default=256)
    ap.add_argument("--lr", type=float, default=0.05)
    args = ap.parse_args()

    df = pd.read_parquet(ROOT / args.data)
    groups = load_groups(ROOT / args.groups)

    if df.empty or not groups or sum(groups) != len(df):
        raise RuntimeError("bad dataset or groups")

    for c in FEATURES + ["label"]:
        if c not in df.columns:
            raise ValueError(f"missing column: {c}")

    df[FEATURES] = df[FEATURES].astype(np.float32)
    df["label"] = df["label"].astype(np.float32)

    df_tr, g_tr, df_va, g_va = split_by_groups(df, groups, args.test_size, args.seed)

    X_tr = df_tr[FEATURES].to_numpy(dtype=np.float32, copy=False)
    y_tr = df_tr["label"].to_numpy(dtype=np.float32, copy=False)
    X_va = df_va[FEATURES].to_numpy(dtype=np.float32, copy=False)
    y_va = df_va["label"].to_numpy(dtype=np.float32, copy=False)

    dtr = xgb.QuantileDMatrix(X_tr, label=y_tr, feature_names=FEATURES, max_bin=args.max_bin)
    dva = xgb.QuantileDMatrix(X_va, label=y_va, feature_names=FEATURES, max_bin=args.max_bin, ref=dtr)
    dtr.set_group(g_tr)
    dva.set_group(g_va)

    params = {
        "device": "cuda",
        "tree_method": "hist",
        "objective": "rank:pairwise",
        "eval_metric": [f"ndcg@{args.ndcg_k}"],
        "grow_policy": "lossguide",
        "max_leaves": int(args.max_leaves),
        "min_child_weight": 1.0,
        "sampling_method": "gradient_based",
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "learning_rate": float(args.lr),
        "lambda": 2.0,
        "alpha": 0.0,
        "gamma": 0.0,
        "random_state": args.seed,
    }

    print("[INFO] xgboost:", xgb.__version__)
    print("[INFO] train rows:", len(df_tr), "groups:", len(g_tr), "avg_group:", (len(df_tr)/max(1,len(g_tr))))
    print("[INFO] val rows:", len(df_va), "groups:", len(g_va), "avg_group:", (len(df_va)/max(1,len(g_va))))
    print("[INFO] params:", {k: params[k] for k in ["device","tree_method","objective","grow_policy","max_leaves","sampling_method","learning_rate"]})

    booster = xgb.train(
        params=params,
        dtrain=dtr,
        num_boost_round=args.rounds,
        evals=[(dtr, "train"), (dva, "val")],
        early_stopping_rounds=args.early_stop,
        verbose_eval=50,
    )

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    booster.save_model(out_path)
    print(f"[OK] saved model -> {out_path}")
    print("[OK] best_iteration:", booster.best_iteration)
    print("[OK] best_score:", booster.best_score)

if __name__ == "__main__":
    main()
