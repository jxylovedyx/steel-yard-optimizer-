from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

FEATURES = ["thickness_mm", "width_mm", "length_mm", "weight_ton", "level", "blocked_cnt"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/datasets/pairwise.csv")
    ap.add_argument("--out", default="models/pairwise_ranker.joblib")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--test-size", type=float, default=0.2)
    args = ap.parse_args()

    data_path = ROOT / args.data
    df = pd.read_csv(data_path)

    X = df[FEATURES].astype(float).values
    y = df["label"].astype(int).values

    X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=args.test_size, random_state=args.seed, stratify=y)

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_va = scaler.transform(X_va)

    clf = SGDClassifier(
        loss="log_loss",
        alpha=1e-5,
        max_iter=1,          
        tol=None,
        random_state=args.seed
    )

    
    for ep in range(1, args.epochs + 1):
        clf.partial_fit(X_tr, y_tr, classes=np.array([0, 1]))
        prob = clf.predict_proba(X_va)[:, 1]
        pred = (prob >= 0.5).astype(int)
        auc = roc_auc_score(y_va, prob)
        acc = accuracy_score(y_va, pred)
        print(f"[EPOCH {ep}] AUC={auc:.4f} ACC={acc:.4f}")

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"scaler": scaler, "model": clf, "features": FEATURES}, out_path)
    print(f"[OK] saved -> {out_path}")


if __name__ == "__main__":
    main()
