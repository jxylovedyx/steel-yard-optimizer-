from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import sys
from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


FEATURES = ["thickness_mm", "width_mm", "length_mm", "weight_ton", "level", "blocked_cnt"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/datasets/ranker.csv")
    ap.add_argument("--out", default="models/ranker.joblib")
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    data_path = ROOT / args.data
    if not data_path.exists():
        raise FileNotFoundError(f"dataset not found: {data_path}")

    df = pd.read_csv(data_path)
    for c in FEATURES + ["label"]:
        if c not in df.columns:
            raise ValueError(f"missing column: {c}")

    X = df[FEATURES].astype(float).values
    y = df["label"].astype(int).values

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed, stratify=y
    )

    
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            solver="adam",
            alpha=1e-4,
            batch_size=2048,
            learning_rate_init=1e-3,
            max_iter=20,
            random_state=args.seed,
            verbose=True
        ))
    ])

    model.fit(X_train, y_train)

    
    prob = model.predict_proba(X_val)[:, 1]
    pred = (prob >= 0.5).astype(int)
    auc = roc_auc_score(y_val, prob) if len(np.unique(y_val)) > 1 else float("nan")
    acc = accuracy_score(y_val, pred)

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "features": FEATURES}, out_path)

    print(f"[OK] saved model -> {out_path}")
    print(f"[VAL] AUC={auc:.4f}  ACC={acc:.4f}  n_val={len(y_val)}")


if __name__ == "__main__":
    main()
