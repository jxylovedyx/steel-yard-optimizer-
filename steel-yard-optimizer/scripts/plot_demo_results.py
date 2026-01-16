from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import Dict, Any, List
import matplotlib.pyplot as plt

def read_metrics(run_dir: Path) -> Dict[str, Any]:
    p = run_dir / "metrics.json"
    return json.loads(p.read_text(encoding="utf-8"))

def kpi(m: Dict[str, Any]) -> Dict[str, float]:
    k = m.get("kpi", {})
    return {
        "makespan_seconds": float(k.get("makespan_seconds", 0.0)),
        "relocation_count": float(k.get("relocation_count", 0.0)),
        "line_starvation_seconds": float(k.get("line_starvation_seconds", 0.0)),
        "crane_utilization": float(k.get("crane_utilization", 0.0)),
    }

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--labels", nargs="+", required=True)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()
    runs = [Path(r) for r in args.runs]
    labels = args.labels
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    ks = [kpi(read_metrics(rd)) for rd in runs]

    # --- KPI bar chart ---
    keys = ["makespan_seconds", "relocation_count", "line_starvation_seconds"]
    legends = ["Makespan(s) ↓", "Relocations ↓", "Starvation(s) ↓"]
    fig = plt.figure()
    x = list(range(len(labels)))
    width = 0.22
    for j, (key, leg) in enumerate(zip(keys, legends)):
        vals = [k[key] for k in ks]
        xx = [i + (j - 1) * width for i in x]
        plt.bar(xx, vals, width=width, label=leg)
    plt.xticks(x, labels)
    plt.ylabel("Value")
    plt.title("KPI Comparison")
    plt.legend()
    plt.tight_layout()
    fig.savefig(outdir / "kpi_compare.png", dpi=180)
    plt.close(fig)

    # --- Pareto scatter ---
    fig = plt.figure()
    xs = [k["makespan_seconds"] for k in ks]
    ys = [k["relocation_count"] for k in ks]
    plt.scatter(xs, ys)
    for xv, yv, lab in zip(xs, ys, labels):
        plt.text(xv, yv, f" {lab}")
    plt.xlabel("Makespan(s) ↓")
    plt.ylabel("Relocations ↓")
    plt.title("Pareto (lower-left better)")
    plt.tight_layout()
    fig.savefig(outdir / "pareto.png", dpi=180)
    plt.close(fig)

    # --- utilization ---
    fig = plt.figure()
    vals = [k["crane_utilization"] for k in ks]
    plt.bar(labels, vals)
    plt.ylim(0, 1.05)
    plt.ylabel("Utilization")
    plt.title("Crane Utilization (higher better)")
    plt.tight_layout()
    fig.savefig(outdir / "utilization.png", dpi=180)
    plt.close(fig)

    print("[OK] saved: kpi_compare.png, pareto.png, utilization.png")

if __name__ == "__main__":
    main()
