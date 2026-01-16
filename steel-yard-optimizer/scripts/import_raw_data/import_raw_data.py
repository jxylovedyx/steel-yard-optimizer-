from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import sys
from pathlib import Path
from typing import Dict, Any, Optional

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts._lib.utils import ensure_dir, write_json, now_iso


def _read_table(raw_dir: Path, name: str) -> Optional[pd.DataFrame]:
    csv_path = raw_dir / f"{name}.csv"
    xlsx_path = raw_dir / f"{name}.xlsx"
    if csv_path.exists():
        return pd.read_csv(csv_path)
    if xlsx_path.exists():
        return pd.read_excel(xlsx_path)
    return None


def _apply_mapping(df: pd.DataFrame, mapping: Dict[str, str]) -> pd.DataFrame:
    # mapping: raw_col -> standard_col
    rename = {k: v for k, v in mapping.items() if k in df.columns}
    df = df.rename(columns=rename)
    return df


def _required_cols_check(df: pd.DataFrame, required: list[str]) -> list[str]:
    missing = [c for c in required if c not in df.columns]
    return missing


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default="data/raw")
    ap.add_argument("--interim-dir", default="data/interim")
    ap.add_argument("--mapping", default="", help="Optional YAML: table-> {raw_col: std_col}")
    ap.add_argument("--report", default="data/reports/data_quality.json")
    args = ap.parse_args()

    raw_dir = ROOT / args.raw_dir
    interim_dir = ROOT / args.interim_dir
    ensure_dir(interim_dir)
    ensure_dir(Path(args.report).parent if Path(args.report).is_absolute() else ROOT / Path(args.report).parent)

    mapping: Dict[str, Any] = {}
    if args.mapping:
        mp = Path(args.mapping)
        if not mp.is_absolute():
            mp = ROOT / mp
        if mp.exists():
            mapping = yaml.safe_load(mp.read_text(encoding="utf-8")) or {}

    # tables we support
    tables = {
        "plates": ["plate_id", "area", "stack_id", "level", "thickness_mm", "width_mm", "length_mm", "weight_ton"],
        "yard_stacks": ["area", "stack_id", "max_levels"],
        "equipment_cranes": ["crane_id", "max_load_ton", "pick_seconds", "place_seconds"],
        "demands": ["cutting_line_id", "plate_id"],
    }

    report = {"created_at": now_iso(), "raw_dir": str(raw_dir), "interim_dir": str(interim_dir), "tables": {}}

    for name, required in tables.items():
        df = _read_table(raw_dir, name)
        if df is None:
            report["tables"][name] = {"status": "missing", "message": "no csv/xlsx found"}
            continue

        # apply mapping if provided
        table_map = mapping.get(name, {}) if isinstance(mapping, dict) else {}
        if isinstance(table_map, dict) and table_map:
            df = _apply_mapping(df, table_map)

        missing = _required_cols_check(df, required)

        # light normalization
        if name == "plates":
            df["plate_id"] = df["plate_id"].astype(str)
            df["area"] = df["area"].astype(str)
            df["stack_id"] = df["stack_id"].astype(str)
            df["level"] = df["level"].astype(int)
        if name == "demands":
            df["plate_id"] = df["plate_id"].astype(str)
            df["cutting_line_id"] = df["cutting_line_id"].astype(str)
            if "seq_no" in df.columns:
                df["seq_no"] = df["seq_no"].astype(int)

        out_path = interim_dir / f"{name}.csv"
        df.to_csv(out_path, index=False)

        report["tables"][name] = {
            "status": "ok" if not missing else "invalid",
            "rows": int(len(df)),
            "cols": int(len(df.columns)),
            "missing_required": missing,
            "written_to": str(out_path),
        }

    # global checks
    global_issues = []
    plates_csv = interim_dir / "plates.csv"
    if plates_csv.exists():
        plates_df = pd.read_csv(plates_csv)
        if "plate_id" in plates_df.columns:
            dup = plates_df["plate_id"][plates_df["plate_id"].duplicated()].tolist()
            if dup:
                global_issues.append({"type": "duplicate_plate_id", "count": len(dup), "examples": dup[:20]})

    report["global_issues"] = global_issues
    out_report = Path(args.report)
    if not out_report.is_absolute():
        out_report = ROOT / out_report
    write_json(out_report, report)
    print(f"[OK] import_raw_data finished. quality report -> {out_report}")


if __name__ == "__main__":
    main()
