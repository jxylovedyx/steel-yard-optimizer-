from __future__ import annotations

import sys
from pathlib import Path

# ensure project root on sys.path (so `import scripts._lib...` works)
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
from pathlib import Path
from typing import List

import pandas as pd

from scripts._lib.utils import ensure_dir, read_csv, write_csv


REQUIRED_RAW = ["plates.csv", "yard_stacks.csv", "equipment_cranes.csv", "demands.csv"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default="data/raw")
    ap.add_argument("--out", default="data/interim")
    args = ap.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out)
    ensure_dir(out_dir)

    missing = [str(raw_dir / f) for f in REQUIRED_RAW if not (raw_dir / f).exists()]
    if missing:
        raise FileNotFoundError(f"Missing raw files: {missing}")

    # plates
    plates = read_csv(raw_dir / "plates.csv")
    # normalize eligible_lines (pipe separated -> list-like string kept; instance builder will split)
    if "eligible_lines" not in plates.columns:
        plates["eligible_lines"] = ""
    if "status" not in plates.columns:
        plates["status"] = "in_yard"

    # yard stacks
    yard = read_csv(raw_dir / "yard_stacks.csv")
    if "max_levels" not in yard.columns:
        yard["max_levels"] = 8

    cranes = read_csv(raw_dir / "equipment_cranes.csv")
    if "seconds_per_meter" not in cranes.columns:
        cranes["seconds_per_meter"] = 0.8

    demands = read_csv(raw_dir / "demands.csv")
    # demands expected columns: cutting_line_id, plate_id, seq_no
    if not {"cutting_line_id", "plate_id"}.issubset(set(demands.columns)):
        raise ValueError("demands.csv must have cutting_line_id, plate_id columns")
    if "seq_no" not in demands.columns:
        demands["seq_no"] = demands.groupby("cutting_line_id").cumcount() + 1

    # write interim
    write_csv(out_dir / "plates.csv", plates)
    write_csv(out_dir / "yard_stacks.csv", yard)
    write_csv(out_dir / "equipment_cranes.csv", cranes)
    write_csv(out_dir / "demands.csv", demands)

    print(f"[OK] interim generated at: {out_dir.resolve()}")
    for f in REQUIRED_RAW:
        p = out_dir / f
        print(f" - {p} ({p.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
