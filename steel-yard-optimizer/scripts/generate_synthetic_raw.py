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
from typing import List, Tuple

import pandas as pd


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


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
    weight = thickness * width * length * 7.85e-9
    weight = max(0.2, weight * rng.uniform(0.9, 1.1))
    return float(thickness), float(width), float(length), float(round(weight, 3))


def generate(
    out_dir: Path,
    areas: int,
    stacks_per_area: int,
    max_levels: int,
    plates: int,
    lines: int,
    demand_per_line: int,
    seed: int,
    auto_capacity: bool,
) -> None:
    rng = random.Random(seed)
    _ensure_dir(out_dir)

    capacity = areas * stacks_per_area * max_levels
    if plates > capacity:
        if not auto_capacity:
            raise ValueError(f"plates({plates}) > capacity({capacity}). Increase stacks_per_area/max_levels/areas or enable --auto-capacity")
        # 自动扩容：优先增 max_levels
        need_levels = (plates + areas * stacks_per_area - 1) // (areas * stacks_per_area)
        max_levels = max(max_levels, int(need_levels))
        capacity = areas * stacks_per_area * max_levels

    area_list = _area_names(areas)

    yard_rows = []
    stack_keys: List[Tuple[str, str]] = []
    for a in area_list:
        for i in range(1, stacks_per_area + 1):
            sid = f"{a}-S{i:03d}"
            stack_keys.append((a, sid))
            yard_rows.append({"area": a, "stack_id": sid, "max_levels": max_levels})
    pd.DataFrame(yard_rows).to_csv(out_dir / "yard_stacks.csv", index=False)

    pd.DataFrame([{
        "crane_id": "C1",
        "max_load_ton": 50,
        "pick_seconds": 25,
        "place_seconds": 25,
        "seconds_per_meter": 0.8
    }]).to_csv(out_dir / "equipment_cranes.csv", index=False)

    slots: List[Tuple[str, str, int]] = []
    for a, sid in stack_keys:
        for lv in range(1, max_levels + 1):
            slots.append((a, sid, lv))
    rng.shuffle(slots)
    chosen = slots[:plates]

    plate_rows = []
    for idx, (a, sid, lv) in enumerate(chosen, start=1):
        pid = f"P{idx:06d}"
        t, w, l, wt = _pick_spec(rng)
        plate_rows.append({
            "plate_id": pid,
            "area": a,
            "stack_id": sid,
            "level": lv,
            "thickness_mm": t,
            "width_mm": w,
            "length_mm": l,
            "weight_ton": wt,
            "project_id": f"PRJ{rng.randint(1, 80):03d}",
            "segment_id": f"SEG{rng.randint(1, 800):04d}",
            "status": "in_yard",
            "eligible_lines": "|".join([f"L{i+1}" for i in range(lines)])
        })
    plates_df = pd.DataFrame(plate_rows)
    plates_df["level"] = plates_df.groupby(["area", "stack_id"])["level"].rank(method="first").astype(int)
    plates_df.to_csv(out_dir / "plates.csv", index=False)

    all_ids = plates_df["plate_id"].tolist()
    rng.shuffle(all_ids)
    total_need = lines * demand_per_line
    if total_need > len(all_ids):
        # 自动收缩需求
        demand_per_line = len(all_ids) // lines
        total_need = lines * demand_per_line

    demand_rows = []
    cursor = 0
    for li in range(1, lines + 1):
        lid = f"L{li}"
        take = all_ids[cursor: cursor + demand_per_line]
        cursor += demand_per_line
        for seq_no, pid in enumerate(take, start=1):
            demand_rows.append({"cutting_line_id": lid, "plate_id": pid, "seq_no": seq_no})
    pd.DataFrame(demand_rows).to_csv(out_dir / "demands.csv", index=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/raw")
    ap.add_argument("--areas", type=int, default=4)
    ap.add_argument("--stacks-per-area", type=int, default=120)
    ap.add_argument("--max-levels", type=int, default=16)
    ap.add_argument("--plates", type=int, default=10000)
    ap.add_argument("--lines", type=int, default=8)
    ap.add_argument("--demand-per-line", type=int, default=1600)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--auto-capacity", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out)
    generate(
        out_dir=out_dir,
        areas=args.areas,
        stacks_per_area=args.stacks_per_area,
        max_levels=args.max_levels,
        plates=args.plates,
        lines=args.lines,
        demand_per_line=args.demand_per_line,
        seed=args.seed,
        auto_capacity=args.auto_capacity,
    )

    print(f"[OK] synthetic raw generated in: {out_dir.resolve()}")
    for f in ["plates.csv", "yard_stacks.csv", "equipment_cranes.csv", "demands.csv"]:
        p = out_dir / f
        print(f" - {p} ({p.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
