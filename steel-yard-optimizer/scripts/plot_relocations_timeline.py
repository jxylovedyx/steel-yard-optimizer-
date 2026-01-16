from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    p = run_dir / "exports" / "crane_tasks.csv"
    if not p.exists():
        raise FileNotFoundError(p)
    df = pd.read_csv(p)

    if "type" not in df.columns or "start_seconds" not in df.columns:
        raise ValueError("crane_tasks.csv missing type/start_seconds")

    df = df.copy()
    df["start_seconds"] = df["start_seconds"].astype(float)
    df = df.sort_values("start_seconds")

    
    if (df["type"] == "relocate").any():
        t = df[df["type"] == "relocate"]["start_seconds"]
    else:
        t = df["start_seconds"]

 
    fig = plt.figure()
    xs = t.tolist()
    ys = list(range(1, len(xs) + 1))
    plt.plot(xs, ys)
    plt.xlabel("Time (s)")
    plt.ylabel("Cumulative relocation tasks")
    plt.title(f"Relocation Timeline ({args.label})")
    plt.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=180)
    plt.close(fig)

if __name__ == "__main__":
    main()
