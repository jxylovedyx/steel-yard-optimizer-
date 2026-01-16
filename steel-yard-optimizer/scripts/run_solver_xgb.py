from __future__ import annotations

import sys
from pathlib import Path

# ensure project root on sys.path (so `import scripts._lib...` works)
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[1]

FEATURES = [
    "thickness_mm","width_mm","length_mm","weight_ton","level","blocked_cnt","is_top",
    "stack_height","stack_count","area_hash","stack_hash",
    "step","demand_pos","remaining_pos","remaining_cnt",
]

def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def read_json(p: Path) -> Any:
    return json.loads(p.read_text(encoding="utf-8"))

def write_json(p: Path, obj: Any) -> None:
    ensure_dir(p.parent)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def gen_run_id(scenario: str, solver: str, exp: str) -> str:
    ts = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    return f"{ts}_{scenario}_{solver}_{exp}"

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

def remove_block_relation(plates: Dict[str, Dict[str, Any]], removed_plate_id: str) -> None:
    for p in plates.values():
        if removed_plate_id in (p.get("blocked_by", []) or []):
            p["blocked_by"] = [x for x in p["blocked_by"] if x != removed_plate_id]

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance", required=True)
    ap.add_argument("--model", default="models/xgb_ranker.json")
    ap.add_argument("--scenario", default="prod")
    ap.add_argument("--exp", default="xgb")
    ap.add_argument("--output-dir", default="data/results")
    args = ap.parse_args()

    inst_path = Path(args.instance)
    if not inst_path.is_absolute():
        inst_path = ROOT / inst_path
    if not inst_path.exists():
        raise FileNotFoundError(f"Instance not found: {inst_path}")

    model_path = Path(args.model)
    if not model_path.is_absolute():
        model_path = ROOT / model_path
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    instance = read_json(inst_path)
    plates_list = instance.get("plates", []) or []
    plates: Dict[str, Dict[str, Any]] = {p["plate_id"]: p for p in plates_list}
    stack_stats = build_stack_stats(plates_list)

    booster = xgb.Booster()
    booster.load_model(str(model_path))
    booster.set_param({"device": "cuda"})

    cranes = (instance.get("equipment", {}) or {}).get("cranes", [])
    crane = cranes[0] if cranes else {"crane_id": "C1", "pick_seconds": 25, "place_seconds": 25, "seconds_per_meter": 0.8}
    pick_s = float(crane.get("pick_seconds", 25))
    place_s = float(crane.get("place_seconds", 25))
    sec_per_m = float(crane.get("seconds_per_meter", 0.8))
    travel_m = 20.0
    move_s = pick_s + place_s + sec_per_m * travel_m

    relocations: List[Dict[str, Any]] = []
    crane_tasks: List[Dict[str, Any]] = []
    line_sequences: List[Dict[str, Any]] = []

    t_crane = 0.0
    t_line_finish: Dict[str, float] = {}

    run_id = gen_run_id(args.scenario, "xgb_ranker", args.exp)

    for d in instance.get("demands", []) or []:
        lid = str(d.get("cutting_line_id", ""))
        items = [pid for pid in (d.get("items", []) or []) if pid in plates]
        demand_pos_map = {pid: i for i, pid in enumerate(items)}
        remaining = list(items)

        seq_out: List[Dict[str, Any]] = []
        t_line = t_line_finish.get(lid, 0.0)
        step = 0

        while remaining:
            step += 1
            X = []
            for pos, pid in enumerate(remaining):
                f = plate_features(plates[pid], stack_stats)
                f.update({
                    "step": float(step),
                    "demand_pos": float(demand_pos_map.get(pid, -1)),
                    "remaining_pos": float(pos),
                    "remaining_cnt": float(len(remaining)),
                })
                X.append([f[k] for k in FEATURES])

            X_np = np.asarray(X, dtype=np.float32)
            dm = xgb.DMatrix(X_np, feature_names=FEATURES)
            scores = booster.predict(dm)
            best_i = int(np.argmax(scores))
            chosen_id = remaining[best_i]

            blockers = list(plates[chosen_id].get("blocked_by", []) or [])
            for b in blockers:
                if b in plates:
                    loc_from = dict(plates[b]["location"])
                    task_id = f"R_{lid}_{chosen_id}_{b}_{len(relocations)+1}"
                    relocations.append({
                        "task_id": task_id,
                        "plate_id": b,
                        "from": loc_from,
                        "to": {"buffer_id": "BUF1"},
                        "reason": "unblock",
                        "estimated_seconds": move_s,
                    })
                    crane_tasks.append({
                        "task_id": task_id,
                        "type": "relocate",
                        "plate_id": b,
                        "crane_id": crane.get("crane_id", "C1"),
                        "start_seconds": t_crane,
                        "finish_seconds": t_crane + move_s,
                    })
                    t_crane += move_s
                    plates[b]["status"] = "reserved"
                    remove_block_relation(plates, b)

            eta = max(t_crane, t_line)
            start = eta
            finish = start + move_s
            t_crane = finish
            t_line = finish

            plates[chosen_id]["status"] = "to_cut"
            seq_out.append({"plate_id": chosen_id, "seq_no": step, "eta_seconds": eta, "start_seconds": start, "finish_seconds": finish})
            crane_tasks.append({
                "task_id": f"OUT_{lid}_{chosen_id}_{step}",
                "type": "outbound",
                "plate_id": chosen_id,
                "crane_id": crane.get("crane_id", "C1"),
                "start_seconds": start,
                "finish_seconds": finish,
            })

            remaining = [pid for pid in remaining if pid != chosen_id]

        t_line_finish[lid] = t_line
        line_sequences.append({"cutting_line_id": lid, "sequence": seq_out})

    plan = {
        "meta": {"run_id": run_id, "instance_id": instance.get("meta", {}).get("instance_id", "unknown"), "solver": "xgb_ranker", "created_at": now_iso()},
        "line_sequences": line_sequences,
        "relocations": relocations,
        "crane_tasks": crane_tasks,
        "warnings": [],
    }

    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_run = out_dir / run_id
    ensure_dir(out_run / "exports")
    ensure_dir(out_run / "logs")

    write_json(out_run / "plan.json", plan)
    write_json(out_run / "input_manifest.json", {
        "run_id": run_id,
        "instance_path": str(inst_path),
        "instance_sha256": sha256_file(inst_path),
        "model_path": str(model_path),
        "created_at": now_iso(),
    })
    print(f"[OK] run_id={run_id}")
    print(f"[OK] plan -> {out_run / 'plan.json'}")

if __name__ == "__main__":
    main()
