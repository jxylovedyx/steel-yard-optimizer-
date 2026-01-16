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

from scripts._lib.utils import now_iso, ensure_dir, gen_run_id, read_json, sha256_file, write_json, dump_yaml
from scripts._lib.config import resolve_config
from scripts._lib.schema import load_schema, validate_json
from scripts._lib.sim import simulate_plan

ROOT = Path(__file__).resolve().parents[1]


def index_plates(instance: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {p["plate_id"]: p for p in instance["plates"]}


def remove_block_relation(plates: Dict[str, Dict[str, Any]], removed_plate_id: str) -> None:
    for p in plates.values():
        if removed_plate_id in (p.get("blocked_by", []) or []):
            p["blocked_by"] = [x for x in p["blocked_by"] if x != removed_plate_id]


def solve_baseline(instance: Dict[str, Any], cfg: Dict[str, Any]) -> Dict[str, Any]:
    plates = index_plates(instance)

    cranes = (instance.get("equipment", {}) or {}).get("cranes", [])
    crane = cranes[0] if cranes else {"crane_id": "C1", "pick_seconds": 25, "place_seconds": 25, "seconds_per_meter": 0.8}
    pick_s = float(crane.get("pick_seconds", 25))
    place_s = float(crane.get("place_seconds", 25))
    sec_per_m = float(crane.get("seconds_per_meter", 0.8))
    travel_m = float(((cfg.get("solver_params", {}) or {}).get("travel_meters_per_move", 20)))
    move_s = pick_s + place_s + sec_per_m * travel_m

    relocations: List[Dict[str, Any]] = []
    crane_tasks: List[Dict[str, Any]] = []
    line_sequences: List[Dict[str, Any]] = []

    t_crane = 0.0
    t_line_finish: Dict[str, float] = {}

    def do_relocate(blocker_id: str, ref_line: str, ref_target: str) -> None:
        nonlocal t_crane
        task_id = f"R_{ref_line}_{ref_target}_{blocker_id}_{len(relocations)+1}"
        relocations.append({
            "task_id": task_id,
            "plate_id": blocker_id,
            "from": dict(plates[blocker_id]["location"]),
            "to": {"buffer_id": "BUF1"},
            "reason": "unblock",
            "estimated_seconds": move_s,
        })
        crane_tasks.append({
            "task_id": task_id,
            "type": "relocate",
            "plate_id": blocker_id,
            "crane_id": crane.get("crane_id", "C1"),
            "start_seconds": t_crane,
            "finish_seconds": t_crane + move_s,
        })
        t_crane += move_s
        plates[blocker_id]["status"] = "reserved"
        remove_block_relation(plates, blocker_id)

    for d in instance.get("demands", []) or []:
        lid = str(d["cutting_line_id"])
        items = [str(x) for x in (d.get("items", []) or [])]
        seq_out: List[Dict[str, Any]] = []
        t_line = t_line_finish.get(lid, 0.0)
        step = 0

        for pid in items:
            if pid not in plates:
                continue
            step += 1
            blockers = list(plates[pid].get("blocked_by", []) or [])
            for b in blockers:
                if b in plates:
                    do_relocate(b, lid, pid)

            eta = max(t_crane, t_line)
            start = eta
            finish = start + move_s
            t_crane = finish
            t_line = finish

            plates[pid]["status"] = "to_cut"
            seq_out.append({"plate_id": pid, "seq_no": step, "eta_seconds": eta, "start_seconds": start, "finish_seconds": finish})
            crane_tasks.append({
                "task_id": f"OUT_{lid}_{pid}_{step}",
                "type": "outbound",
                "plate_id": pid,
                "crane_id": crane.get("crane_id", "C1"),
                "start_seconds": start,
                "finish_seconds": finish,
            })

        t_line_finish[lid] = t_line
        line_sequences.append({"cutting_line_id": lid, "sequence": seq_out})

    plan = {
        "meta": {"run_id": "TO_BE_FILLED", "instance_id": instance["meta"]["instance_id"], "solver": "baseline_rule", "created_at": now_iso()},
        "line_sequences": line_sequences,
        "relocations": relocations,
        "crane_tasks": crane_tasks,
        "warnings": [],
    }
    return plan


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/baseline.yaml")
    ap.add_argument("--instance", required=True)
    args = ap.parse_args()

    cfg = resolve_config(args.config)
    scenario = cfg.get("scenario", "toy")
    solver = "baseline_rule"
    exp = cfg.get("experiment", "exp")
    run_id = gen_run_id(scenario, solver, exp)

    inst_path = Path(args.instance)
    if not inst_path.is_absolute():
        inst_path = ROOT / inst_path
    if not inst_path.exists():
        raise FileNotFoundError(f"Instance not found: {inst_path}")

    instance = read_json(inst_path)
    validate_json(instance, load_schema(ROOT / "schemas" / "instance.schema.json"), label="instance")

    plan = solve_baseline(instance, cfg)
    plan["meta"]["run_id"] = run_id
    validate_json(plan, load_schema(ROOT / "schemas" / "plan.schema.json"), label="plan")

    out_base = Path(((cfg.get("io", {}) or {}).get("output_dir", "data/results")))
    if not out_base.is_absolute():
        out_base = ROOT / out_base
    out_dir = out_base / run_id
    ensure_dir(out_dir / "exports")
    ensure_dir(out_dir / "logs")

    write_json(out_dir / "plan.json", plan)
    dump_yaml(out_dir / "config_snapshot.yaml", cfg)
    write_json(out_dir / "input_manifest.json", {
        "run_id": run_id,
        "instance_path": str(inst_path),
        "instance_sha256": sha256_file(inst_path),
        "created_at": now_iso(),
    })

    kpi = simulate_plan(instance, plan, cfg)
    metrics = {"meta": {"run_id": run_id, "instance_id": instance["meta"]["instance_id"], "created_at": now_iso()}, "kpi": kpi}
    validate_json(metrics, load_schema(ROOT / "schemas" / "metrics.schema.json"), label="metrics")
    write_json(out_dir / "metrics.json", metrics)

    print(f"[OK] run_id={run_id}")
    print(f"[OK] plan -> {out_dir / 'plan.json'}")
    print(f"[OK] metrics -> {out_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
