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
from typing import Any, Dict, List

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


FEATURES = ["thickness_mm", "width_mm", "length_mm", "weight_ton", "level", "blocked_cnt"]


def read_json(p: Path) -> Any:
    return json.loads(p.read_text(encoding="utf-8"))


def feat(plate: Dict[str, Any]) -> Dict[str, float]:
    spec = plate.get("spec", {}) or {}
    loc = plate.get("location", {}) or {}
    blocked = plate.get("blocked_by", []) or []
    return {
        "thickness_mm": float(spec.get("thickness_mm", 0.0)),
        "width_mm": float(spec.get("width_mm", 0.0)),
        "length_mm": float(spec.get("length_mm", 0.0)),
        "weight_ton": float(spec.get("weight_ton", 0.0)),
        "level": float(loc.get("level", 0.0)),
        "blocked_cnt": float(len(blocked)),
    }


def diff(a: Dict[str, float], b: Dict[str, float]) -> Dict[str, float]:
    return {k: float(a[k]) - float(b[k]) for k in FEATURES}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="data/results")
    ap.add_argument("--out", default="data/datasets/pairwise.csv")
    ap.add_argument("--neg-k", type=int, default=20)
    ap.add_argument("--max-runs", type=int, default=200)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    results_dir = ROOT / args.results_dir
    run_dirs = sorted([p for p in results_dir.iterdir() if p.is_dir()], key=lambda p: p.stat().st_mtime, reverse=True)[: args.max_runs]

    rows: List[Dict[str, Any]] = []
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

        inst = read_json(inst_path)
        plan = read_json(plan_p)

        plates = {p["plate_id"]: p for p in inst.get("plates", [])}
        all_ids = list(plates.keys())

        for ls in plan.get("line_sequences", []):
            lid = ls.get("cutting_line_id", "")
            seq = ls.get("sequence", []) or []

           
            demand_map = {}
            for d in inst.get("demands", []):
                if d.get("cutting_line_id") == lid:
                    demand_map = {pid: i for i, pid in enumerate(d.get("items", []) or [])}
                    break
            cand_pool = [pid for pid in demand_map.keys() if pid in plates]
            if not cand_pool:
                cand_pool = [pid for pid in all_ids]

            for step, chosen in enumerate(seq, start=1):
                chosen_id = chosen.get("plate_id")
                if not chosen_id or chosen_id not in plates:
                    continue
                c = feat(plates[chosen_id])

               
                scored = []
                for pid2 in cand_pool:
                    if pid2 == chosen_id:
                        continue
                    f2 = feat(plates[pid2])
                    s = abs(f2["level"] - c["level"]) + 2.0 * abs(f2["blocked_cnt"] - c["blocked_cnt"])
                    scored.append((s, pid2))
                scored.sort(key=lambda x: x[0])
                hard = [pid2 for _, pid2 in scored[: max(200, args.neg_k * 10)]]
                rng.shuffle(hard)
                neg_ids = hard[: args.neg_k]

                for neg_id in neg_ids:
                    n = feat(plates[neg_id])
                    dcn = diff(c, n)
                    dnc = diff(n, c)
                    rows.append({"label": 1, "line_id": lid, "step": step, **dcn})
                    rows.append({"label": 0, "line_id": lid, "step": step, **dnc})

        used += 1

    if not rows:
        raise RuntimeError("No pairwise rows generated. Ensure you have runs with plan.json and instance_path exists.")

    df = pd.DataFrame(rows)
    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    pos = int((df["label"] == 1).sum())
    neg = int((df["label"] == 0).sum())
    print(f"[OK] dataset -> {out_path}  rows={len(df)} pos={pos} neg={neg}")


if __name__ == "__main__":
    main()
