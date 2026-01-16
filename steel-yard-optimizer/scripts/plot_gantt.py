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
    p = run_dir / "exports" / "line_sequence.csv"
    if not p.exists():
        raise FileNotFoundError(p)
    df = pd.read_csv(p)

    required = {"cutting_line_id", "start_seconds", "finish_seconds", "plate_id"}
    if not required.issubset(df.columns):
        raise ValueError(f"line_sequence.csv missing columns: {required - set(df.columns)}")

    df = df.copy()
    df["cutting_line_id"] = df["cutting_line_id"].astype(str)
    df["start_seconds"] = df["start_seconds"].astype(float)
    df["finish_seconds"] = df["finish_seconds"].astype(float)
    df["dur"] = df["finish_seconds"] - df["start_seconds"]

    lines = sorted(df["cutting_line_id"].unique().tolist())
    ymap = {lid: i for i, lid in enumerate(lines)}

    fig = plt.figure(figsize=(10, max(3, 0.6 * len(lines))))
    for lid in lines:
        g = df[df["cutting_line_id"] == lid].sort_values("start_seconds")
        y = ymap[lid]
        # 用条形图画每个plate的加工区间
        plt.barh([y]*len(g), g["dur"].tolist(), left=g["start_seconds"].tolist(), height=0.35)
    plt.yticks(list(ymap.values()), lines)
    plt.xlabel("Time (s)")
    plt.title(f"Gantt: Cutting Lines ({args.label})")
    plt.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=180)
    plt.close(fig)

if __name__ == "__main__":
    main()
