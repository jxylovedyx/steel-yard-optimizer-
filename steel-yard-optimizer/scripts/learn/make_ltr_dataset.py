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

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def read_json(p: Path) -> Any:
    return json.loads(p.read_text(encoding="utf-8"))


def stable_hash(s: str) -> int:
    h = 2166136261
    for ch in s:
        h ^= ord(ch)
        h = (h * 16777619) & 0xFFFFFFFF
    return int(h)


def build_stack_stats(plates: List[Dict[str, Any]]) -> Dict[Tuple[str, str], Dict[str, float]]:
    stats: Dict[Tuple[str, str], Dict[str, float]] = {}
    for p in plates:
        loc = p.get("location", {}) or {}
        area = str(loc.get("area", ""))
        sid = str(loc.get("stack_id", ""))
        lv = float(loc.get("level", 0.0))
        key = (area, sid)
        if key not in stats:
            stats[key] = {"height": 0.0, "count": 0.0}
        stats[key]["height"] = max(stats[key]["height"], lv)
        stats[key]["count"] += 1.0
    return stats


def plate_features(p: Dict[str, Any], stack_stats: Dict[Tuple[str, str], Dict[str, float]]) -> Dict[str, float]:
    spec = p.get("spec", {}) or {}
    loc = p.get("location", {}) or {}
    blocked = p.get("blocked_by", []) or []
    area = str(loc.get("area", ""))
    sid = str(loc.get("stack_id", ""))
    lv = float(loc.get("level", 0.0))
    st = stack_stats.get((area, sid), {"height": 0.0, "count": 0.0})
    return {
        "thickness_mm": float(spec.get("thickness_mm", 0.0)),
        "width_mm": float(spec.get("width_mm", 0.0)),
        "length_mm": float(spec.get("length_mm", 0.0)),
        "weight_ton": float(spec.get("weight_ton", 0.0)),
        "level": float(lv),
        "blocked_cnt": float(len(blocked)),
        "is_top": 1.0 if len(blocked) == 0 else 0.0,
        "stack_height": float(st.get("height", 0.0)),
        "stack_count": float(st.get("count", 0.0)),
        "area_hash": float(stable_hash(area) % 1024),
        "stack_hash": float(stable_hash(f"{area}:{sid}") % 65536),
    }


def demand_index_map(instance: Dict[str, Any]) -> Dict[str, Dict[str, int]]:
    out: Dict[str, Dict[str, int]] = {}
    for d in instance.get("demands", []) or []:
        lid = str(d.get("cutting_line_id", ""))
        items = d.get("items", []) or []
        out[lid] = {str(pid): i for i, pid in enumerate(items)}
    return out


def pick_candidates(
    rng: random.Random,
    remaining: List[str],
    chosen_id: str,
    plates: Dict[str, Dict[str, Any]],
    stack_stats: Dict[Tuple[str, str], Dict[str, float]],
    max_candidates: int,
) -> List[str]:
    if len(remaining) <= max_candidates:
        return list(remaining)

    chosen = plates[chosen_id]
    cf = plate_features(chosen, stack_stats)
    c_level = cf["level"]
    c_blocked = cf["blocked_cnt"]

    scored = []
    for pid in remaining:
        if pid == chosen_id:
            continue
        pf = plate_features(plates[pid], stack_stats)
        s = abs(pf["level"] - c_level) + 2.0 * abs(pf["blocked_cnt"] - c_blocked) - 0.5 * pf["is_top"]
        scored.append((s, pid))
    scored.sort(key=lambda x: x[0])

    keep = [chosen_id]
    for _, pid in scored[: max_candidates // 2]:
        keep.append(pid)

    pool = [pid for pid in remaining if pid not in set(keep)]
    rng.shuffle(pool)
    need = max_candidates - len(keep)
    keep.extend(pool[:need])
    return keep


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="data/results")
    ap.add_argument("--out", default="data/datasets/ltr.parquet")
    ap.add_argument("--max-runs", type=int, default=500)
    ap.add_argument("--max-candidates", type=int, default=128)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    results_dir = ROOT / args.results_dir
    if not results_dir.exists():
        raise FileNotFoundError(results_dir)

    run_dirs = sorted([p for p in results_dir.iterdir() if p.is_dir()],
                      key=lambda p: p.stat().st_mtime, reverse=True)[: args.max_runs]

    rows: List[Dict[str, Any]] = []
    groups: List[int] = []

    for rd in run_dirs:
        plan_p = rd / "plan.json"
        manifest_p = rd / "input_manifest.json"
        if not plan_p.exists() or not manifest_p.exists():
            continue

        manifest = read_json(manifest_p)
        inst_path = Path(str(manifest.get("instance_path", "")))
        if not inst_path.is_absolute():
            inst_path = ROOT / inst_path
        if not inst_path.exists():
            continue

        instance = read_json(inst_path)
        plan = read_json(plan_p)

        plates_list = instance.get("plates", []) or []
        plates = {str(p["plate_id"]): p for p in plates_list}
        stack_stats = build_stack_stats(plates_list)
        dmap = demand_index_map(instance)

        for ls in plan.get("line_sequences", []) or []:
            lid = str(ls.get("cutting_line_id", ""))
            seq = ls.get("sequence", []) or []
            if lid not in dmap:
                continue

            remaining = [pid for pid in dmap[lid].keys() if pid in plates]

            for step, act in enumerate(seq, start=1):
                chosen_id = str(act.get("plate_id", ""))
                if chosen_id not in plates or chosen_id not in remaining:
                    continue

                cand = pick_candidates(rng, remaining, chosen_id, plates, stack_stats, args.max_candidates)

                
                rng.shuffle(cand)

                gcount = 0
                for pos, pid in enumerate(cand):
                    f = plate_features(plates[pid], stack_stats)
                    label = 1.0 if pid == chosen_id else 0.0
                    rows.append({
                        "group_id": f"{rd.name}:{lid}:{step}",
                        "label": label,
                        "line_id": lid,
                        "step": float(step),
                        "demand_pos": float(dmap[lid].get(pid, -1)),
                        "remaining_pos": float(remaining.index(pid)) if pid in remaining else -1.0,
                        "remaining_cnt": float(len(remaining)),
                        **f
                    })
                    gcount += 1

                groups.append(gcount)
                remaining = [pid for pid in remaining if pid != chosen_id]

    if not rows:
        raise RuntimeError("No LTR rows built. Ensure you have plan.json + input_manifest.json with line_sequences.")

    df = pd.DataFrame(rows)
    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)

    group_path = out_path.with_suffix(".groups.txt")
    group_path.write_text("\n".join(str(x) for x in groups), encoding="utf-8")

    pos = int((df["label"] == 1).sum())
    neg = int((df["label"] == 0).sum())
    print(f"[OK] LTR dataset -> {out_path}")
    print(f"[OK] groups -> {group_path}")
    print(f"rows={len(df)} pos={pos} neg={neg} groups={len(groups)} avg_group={len(df)/len(groups):.1f}")


if __name__ == "__main__":
    main()
