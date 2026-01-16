from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def read_json(p: Path) -> Any:
    return json.loads(p.read_text(encoding="utf-8"))


def plate_feature_row(p: Dict[str, Any]) -> Dict[str, float]:
    spec = p.get("spec", {}) or {}
    loc = p.get("location", {}) or {}
    blocked_by = p.get("blocked_by", []) or []
    return {
        "thickness_mm": float(spec.get("thickness_mm", 0.0)),
        "width_mm": float(spec.get("width_mm", 0.0)),
        "length_mm": float(spec.get("length_mm", 0.0)),
        "weight_ton": float(spec.get("weight_ton", 0.0)),
        "level": float(loc.get("level", 0.0)),
        "blocked_cnt": float(len(blocked_by)),
    }


def build_examples_from_one(instance: Dict[str, Any], plan: Dict[str, Any], neg_k: int, seed: int) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    plates = {p["plate_id"]: p for p in instance.get("plates", [])}

    
    all_plate_ids = list(plates.keys())

    rows: List[Dict[str, Any]] = []
    for ls in plan.get("line_sequences", []):
        line_id = ls.get("cutting_line_id", "")
      
        for step, chosen in enumerate(seq, start=1):
            chosen_id = chosen.get("plate_id")
            if not chosen_id or chosen_id not in plates:
                continue

            
            feat = plate_feature_row(plates[chosen_id])
            rows.append({
                "line_id": line_id,
                "step": step,
                "plate_id": chosen_id,
                "label": 1,
                **feat
            })

           
            neg_pool = [pid for pid in all_plate_ids if pid != chosen_id]
            rng.shuffle(neg_pool)
            for pid in neg_pool[:neg_k]:
                feat_n = plate_feature_row(plates[pid])
                rows.append({
                    "line_id": line_id,
                    "step": step,
                    "plate_id": pid,
                    "label": 0,
                    **feat_n
                })

    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="data/results", help="where run_id folders exist")
    ap.add_argument("--out", default="data/datasets/ranker.csv")
    ap.add_argument("--neg-k", type=int, default=20, help="negatives per positive")
    ap.add_argument("--max-runs", type=int, default=200, help="cap number of runs to read")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    results_dir = ROOT / args.results_dir
    if not results_dir.exists():
        raise FileNotFoundError(f"results-dir not found: {results_dir}")

    run_dirs = sorted([p for p in results_dir.iterdir() if p.is_dir()], key=lambda p: p.stat().st_mtime, reverse=True)
    run_dirs = run_dirs[: args.max_runs]

    all_rows: List[Dict[str, Any]] = []
    used = 0

    for rd in run_dirs:
        plan_p = rd / "plan.json"
        manifest_p = rd / "input_manifest.json"
        if not plan_p.exists() or not manifest_p.exists():
            continue

        manifest = read_json(manifest_p)
        inst_path = Path(manifest.get("instance_path", ""))
        if not inst_path.is_absolute():
            inst_path = ROOT / inst_path
        if not inst_path.exists():
            continue

        instance = read_json(inst_path)
        plan = read_json(plan_p)

        rows = build_examples_from_one(instance, plan, neg_k=args.neg_k, seed=args.seed + used)
        if rows:
            all_rows.extend(rows)
            used += 1

    if not all_rows:
        raise RuntimeError("No training rows generated. Ensure you have runs with plan.json + input_manifest.json and instance_path exists.")

    df = pd.DataFrame(all_rows)
    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    pos = int((df["label"] == 1).sum())
    neg = int((df["label"] == 0).sum())
    print(f"[OK] wrote dataset: {out_path}")
    print(f"rows={len(df)} pos={pos} neg={neg} neg_per_pos≈{neg/ max(pos,1):.1f}")


if __name__ == "__main__":
    main()
