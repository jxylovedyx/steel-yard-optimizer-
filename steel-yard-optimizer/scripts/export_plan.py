from __future__ import annotations

import sys
from pathlib import Path

# ensure project root on sys.path (so `import scripts._lib...` works)
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from scripts._lib.utils import ensure_dir, read_json, write_csv

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True)
    ap.add_argument("--out", default="data/exports")
    args = ap.parse_args()

    plan_path = Path(args.plan)
    if not plan_path.is_absolute():
        plan_path = ROOT / plan_path
    if not plan_path.exists():
        raise FileNotFoundError(plan_path)

    out_dir = Path(args.out)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    ensure_dir(out_dir)

    plan = read_json(plan_path)

    # line sequence
    rows = []
    for ls in plan.get("line_sequences", []) or []:
        lid = str(ls["cutting_line_id"])
        for s in ls.get("sequence", []) or []:
            rows.append({
                "cutting_line_id": lid,
                "seq_no": int(s["seq_no"]),
                "plate_id": str(s["plate_id"]),
                "eta_seconds": float(s["eta_seconds"]),
                "start_seconds": float(s["start_seconds"]),
                "finish_seconds": float(s["finish_seconds"])
            })
    df_seq = pd.DataFrame(rows).sort_values(["cutting_line_id", "seq_no"]) if rows else pd.DataFrame(columns=["cutting_line_id","seq_no","plate_id","eta_seconds","start_seconds","finish_seconds"])
    write_csv(out_dir / "line_sequence.csv", df_seq)

    # crane tasks
    df_tasks = pd.DataFrame(plan.get("crane_tasks", []) or [])
    if not df_tasks.empty and "start_seconds" in df_tasks.columns:
        df_tasks = df_tasks.sort_values(["start_seconds"])
    write_csv(out_dir / "crane_tasks.csv", df_tasks if not df_tasks.empty else pd.DataFrame(columns=["task_id","type","plate_id","crane_id","start_seconds","finish_seconds"]))

    # relocations
    df_reloc = pd.DataFrame(plan.get("relocations", []) or [])
    write_csv(out_dir / "relocations.csv", df_reloc if not df_reloc.empty else pd.DataFrame(columns=["task_id","plate_id","reason","estimated_seconds"]))

    print(f"[OK] exports -> {out_dir.resolve()}")
    for f in ["line_sequence.csv", "crane_tasks.csv", "relocations.csv"]:
        p = out_dir / f
        print(f" - {p} ({p.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
