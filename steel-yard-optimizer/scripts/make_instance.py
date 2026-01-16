from __future__ import annotations

import sys
from pathlib import Path

# ensure project root on sys.path (so `import scripts._lib...` works)
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

from scripts._lib.utils import ensure_dir, now_iso, read_csv, write_json
from scripts._lib.schema import load_schema, validate_json


ROOT = Path(__file__).resolve().parents[1]


def _area_names(n: int) -> List[str]:
    names: List[str] = []
    i = 0
    while len(names) < n:
        x = i
        s = ""
        while True:
            s = chr(ord("A") + (x % 26)) + s
            x = x // 26 - 1
            if x < 0:
                break
        names.append(s)
        i += 1
    return names


def _pick_spec(rng: random.Random) -> Tuple[float, float, float, float]:
    thickness = rng.choice([6, 8, 10, 12, 14, 16, 18, 20, 25, 30, 40, 50])
    width = rng.choice([1500, 1800, 2000, 2200, 2500, 2800, 3000])
    length = rng.choice([6000, 8000, 9000, 10000, 12000, 14000, 16000])
    weight = thickness * width * length * 7.85e-9  # tons (rough)
    weight = max(0.2, weight * rng.uniform(0.9, 1.1))
    return float(thickness), float(width), float(length), float(round(weight, 3))


def build_instance_from_interim(interim_dir: Path, scenario: str) -> Dict[str, Any]:
    required = ["plates.csv", "yard_stacks.csv", "equipment_cranes.csv", "demands.csv"]
    missing = [str(interim_dir / f) for f in required if not (interim_dir / f).exists()]
    if missing:
        raise FileNotFoundError(f"Missing interim files: {missing}")

    plates_df = read_csv(interim_dir / "plates.csv")
    yard_df = read_csv(interim_dir / "yard_stacks.csv")
    cranes_df = read_csv(interim_dir / "equipment_cranes.csv")
    dem_df = read_csv(interim_dir / "demands.csv")

    yard = {
        "stacks": [
            {
                "area": str(r["area"]),
                "stack_id": str(r["stack_id"]),
                "max_levels": int(r.get("max_levels", 8)),
                "x": float(r.get("x", 0.0)) if "x" in yard_df.columns else 0.0,
                "y": float(r.get("y", 0.0)) if "y" in yard_df.columns else 0.0
            }
            for _, r in yard_df.iterrows()
        ]
    }

    cranes = []
    for _, r in cranes_df.iterrows():
        cranes.append({
            "crane_id": str(r.get("crane_id", "C1")),
            "max_load_ton": float(r.get("max_load_ton", 50)),
            "pick_seconds": float(r.get("pick_seconds", 25)),
            "place_seconds": float(r.get("place_seconds", 25)),
            "seconds_per_meter": float(r.get("seconds_per_meter", 0.8))
        })

    # build plate blocked_by from levels per stack (LIFO)
    plates: List[Dict[str, Any]] = []
    by_stack: Dict[Tuple[str, str], List[Tuple[int, str]]] = {}
    for _, r in plates_df.iterrows():
        area = str(r["area"])
        sid = str(r["stack_id"])
        lv = int(r["level"])
        pid = str(r["plate_id"])
        by_stack.setdefault((area, sid), []).append((lv, pid))

    above_map: Dict[str, List[str]] = {}
    for key, lst in by_stack.items():
        lst_sorted = sorted(lst, key=lambda x: x[0])
        # higher level blocks lower level
        for i, (lv, pid) in enumerate(lst_sorted):
            blockers = [p2 for (lv2, p2) in lst_sorted if lv2 > lv]
            above_map[pid] = blockers

    for _, r in plates_df.iterrows():
        pid = str(r["plate_id"])
        eligible = str(r.get("eligible_lines", "")).strip()
        eligible_lines = [x for x in eligible.split("|") if x] if eligible else []
        plates.append({
            "plate_id": pid,
            "spec": {
                "thickness_mm": float(r.get("thickness_mm", 0)),
                "width_mm": float(r.get("width_mm", 0)),
                "length_mm": float(r.get("length_mm", 0)),
                "weight_ton": float(r.get("weight_ton", 0)),
            },
            "location": {
                "area": str(r["area"]),
                "stack_id": str(r["stack_id"]),
                "level": int(r["level"]),
            },
            "blocked_by": above_map.get(pid, []),
            "status": str(r.get("status", "in_yard")),
            "eligible_lines": eligible_lines
        })

    # demands -> list per line ordered by seq_no
    dem_df = dem_df.sort_values(["cutting_line_id", "seq_no"])
    demands = []
    for lid, g in dem_df.groupby("cutting_line_id"):
        demands.append({
            "cutting_line_id": str(lid),
            "items": [str(x) for x in g["plate_id"].tolist()]
        })

    inst = {
        "meta": {
            "instance_id": f"{scenario}_interim",
            "scenario": scenario,
            "created_at": now_iso()
        },
        "yard": yard,
        "equipment": {
            "cranes": cranes,
            "travel_model": {"seconds_per_meter": float(cranes[0].get("seconds_per_meter", 0.8)) if cranes else 0.8}
        },
        "plates": plates,
        "demands": demands
    }
    return inst


def build_toy_instance(seed: int, scenario: str, areas: int, stacks_per_area: int, max_levels: int, plates: int, lines: int, demand_per_line: int) -> Dict[str, Any]:
    rng = random.Random(seed)
    capacity = areas * stacks_per_area * max_levels
    if plates > capacity:
        raise ValueError(f"plates({plates}) > capacity({capacity})")

    area_list = _area_names(areas)
    stacks = []
    stack_keys: List[Tuple[str, str, int]] = []
    for a in area_list:
        for i in range(1, stacks_per_area + 1):
            sid = f"{a}-S{i:03d}"
            stacks.append({"area": a, "stack_id": sid, "max_levels": max_levels})
            for lv in range(1, max_levels + 1):
                stack_keys.append((a, sid, lv))

    rng.shuffle(stack_keys)
    chosen = stack_keys[:plates]

    # blocked_by computed after assigning
    plates_raw = []
    for idx, (a, sid, lv) in enumerate(chosen, start=1):
        pid = f"P{idx:06d}"
        t, w, l, wt = _pick_spec(rng)
        plates_raw.append({
            "plate_id": pid,
            "area": a,
            "stack_id": sid,
            "level": lv,
            "thickness_mm": t,
            "width_mm": w,
            "length_mm": l,
            "weight_ton": wt
        })
    df = pd.DataFrame(plates_raw)
    df["level"] = df.groupby(["area", "stack_id"])["level"].rank(method="first").astype(int)

    by_stack: Dict[Tuple[str, str], List[Tuple[int, str]]] = {}
    for _, r in df.iterrows():
        by_stack.setdefault((r["area"], r["stack_id"]), []).append((int(r["level"]), str(r["plate_id"])))
    above_map: Dict[str, List[str]] = {}
    for key, lst in by_stack.items():
        lst_sorted = sorted(lst, key=lambda x: x[0])
        for lv, pid in lst_sorted:
            above_map[pid] = [p2 for (lv2, p2) in lst_sorted if lv2 > lv]

    plate_objs = []
    for _, r in df.iterrows():
        pid = str(r["plate_id"])
        plate_objs.append({
            "plate_id": pid,
            "spec": {
                "thickness_mm": float(r["thickness_mm"]),
                "width_mm": float(r["width_mm"]),
                "length_mm": float(r["length_mm"]),
                "weight_ton": float(r["weight_ton"]),
            },
            "location": {"area": str(r["area"]), "stack_id": str(r["stack_id"]), "level": int(r["level"])},
            "blocked_by": above_map.get(pid, []),
            "status": "in_yard",
            "eligible_lines": [f"L{i}" for i in range(1, lines + 1)]
        })

    all_ids = [p["plate_id"] for p in plate_objs]
    rng.shuffle(all_ids)
    need = lines * demand_per_line
    need = lines * demand_per_line
    if need > len(all_ids):
        # shrink per-line demand to fit available plates (toy/synthetic safety)
        demand_per_line = max(1, len(all_ids) // max(1, lines))
        need = lines * demand_per_line
    for li in range(1, lines + 1):
        lid = f"L{li}"
        take = all_ids[cur:cur + demand_per_line]
        cur += demand_per_line
        demands.append({"cutting_line_id": lid, "items": take})

    inst = {
        "meta": {"instance_id": f"{scenario}_toy", "scenario": scenario, "created_at": now_iso()},
        "yard": {"stacks": stacks},
        "equipment": {
            "cranes": [{"crane_id": "C1", "max_load_ton": 50, "pick_seconds": 25, "place_seconds": 25, "seconds_per_meter": 0.8}],
            "travel_model": {"seconds_per_meter": 0.8}
        },
        "plates": plate_objs,
        "demands": demands
    }
    return inst


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["toy", "interim"], default="toy")
    ap.add_argument("--config", default="configs/baseline.yaml")
    ap.add_argument("--interim-dir", default="data/interim")
    ap.add_argument("--out", default="data/instances/toy_instance.json")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--areas", type=int, default=2)
    ap.add_argument("--stacks-per-area", type=int, default=50)
    ap.add_argument("--max-levels", type=int, default=8)
    ap.add_argument("--plates", type=int, default=600)
    ap.add_argument("--lines", type=int, default=4)
    ap.add_argument("--demand-per-line", type=int, default=300)
    args = ap.parse_args()

    scenario = "toy"
    if args.mode == "interim":
        scenario = "interim"

    if args.mode == "toy":
        inst = build_toy_instance(
            seed=args.seed,
            scenario=scenario,
            areas=args.areas,
            stacks_per_area=args.stacks_per_area,
            max_levels=args.max_levels,
            plates=args.plates,
            lines=args.lines,
            demand_per_line=args.demand_per_line,
        )
    else:
        inst = build_instance_from_interim(ROOT / args.interim_dir, scenario=scenario)

    validate_json(inst, load_schema(ROOT / "schemas" / "instance.schema.json"), label="instance")
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = ROOT / out_path
    ensure_dir(out_path.parent)
    write_json(out_path, inst)
    print(f"[OK] instance -> {out_path.resolve()}")


if __name__ == "__main__":
    main()
