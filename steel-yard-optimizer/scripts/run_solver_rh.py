from __future__ import annotations

import sys
from pathlib import Path

# ensure project root on sys.path (so `import scripts._lib...` works)
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple

from scripts._lib.config import resolve_config
from scripts._lib.schema import load_schema, validate_json
from scripts._lib.sim import simulate_plan
from scripts._lib.utils import now_iso, ensure_dir, gen_run_id, read_json, sha256_file, write_json, dump_yaml

ROOT = Path(__file__).resolve().parents[1]


def index_plates(instance: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {p["plate_id"]: p for p in instance["plates"]}


def remove_block_relation(plates: Dict[str, Dict[str, Any]], removed_plate_id: str) -> None:
    for p in plates.values():
        if removed_plate_id in (p.get("blocked_by", []) or []):
            p["blocked_by"] = [x for x in p["blocked_by"] if x != removed_plate_id]


def move_seconds(instance: Dict[str, Any], cfg: Dict[str, Any]) -> float:
    cranes = (instance.get("equipment", {}) or {}).get("cranes", [])
    crane = cranes[0] if cranes else {"pick_seconds": 25, "place_seconds": 25, "seconds_per_meter": 0.8}
    pick_s = float(crane.get("pick_seconds", 25))
    place_s = float(crane.get("place_seconds", 25))
    sec_per_m = float(crane.get("seconds_per_meter", 0.8))
    travel_m = float(((cfg.get("solver_params", {}) or {}).get("travel_meters_per_move", 20)))
    return pick_s + place_s + sec_per_m * travel_m


def cut_seconds(cfg: Dict[str, Any]) -> float:
    return float(((cfg.get("solver_params", {}) or {}).get("cut_seconds_per_plate", 45)))


def estimate_relocations_for_plate(plates: Dict[str, Dict[str, Any]], pid: str) -> int:
    return int(len(plates[pid].get("blocked_by", []) or []))


def choose_next_action(
    plates: Dict[str, Dict[str, Any]],
    line_remaining: Dict[str, List[str]],
    line_ready: Dict[str, float],
    t_crane: float,
    cfg: Dict[str, Any],
    window_per_line: int
) -> Tuple[str, str]:
    """
    双目标(词典序)的一个工程化近似：
    - 先尽量减少“下一步导致的 makespan 增量”（等价：优先喂当前最饥饿的线）
    - 在候选差不多时，选 relocations 少的
    """
    # 选最饥饿线：ready最小（最早会缺料/最闲）
    lines = [lid for lid, rem in line_remaining.items() if rem]
    if not lines:
        raise RuntimeError("no remaining demand")
    lines.sort(key=lambda lid: line_ready[lid])

    best_lid = None
    best_pid = None
    best_key = None

    for lid in lines[: max(1, len(lines))]:
        window = line_remaining[lid][:window_per_line]
        # 只在窗口内找一个“当前代价最小”的板
        # makespan近似：选择板会产生 reloc_move + outbound_move 的 crane 占用
        for pid in window:
            if pid not in plates:
                continue
            reloc = estimate_relocations_for_plate(plates, pid)
            # 词典序 key：先最小化 line starvation 风险（用 line_ready 和 t_crane 差近似）
            # 其次最小化 reloc
            starvation_risk = max(0.0, t_crane - line_ready[lid])
            key = (starvation_risk, reloc, lid, pid)
            if best_key is None or key < best_key:
                best_key = key
                best_lid = lid
                best_pid = pid

    assert best_lid is not None and best_pid is not None
    return best_lid, best_pid


def solve_rh(instance: Dict[str, Any], cfg: Dict[str, Any]) -> Dict[str, Any]:
    plates = index_plates(instance)
    move_s = move_seconds(instance, cfg)
    cut_s = cut_seconds(cfg)

    rh_cfg = cfg.get("rh", {}) or {}
    W = int(rh_cfg.get("window_per_line", 60))

    # 初始化：每条线 remaining + ready_time
    line_remaining: Dict[str, List[str]] = {}
    line_ready: Dict[str, float] = {}
    for d in instance.get("demands", []) or []:
        lid = str(d["cutting_line_id"])
        items = [str(x) for x in (d.get("items", []) or []) if str(x) in plates]
        line_remaining[lid] = items
        line_ready[lid] = 0.0

    cranes = (instance.get("equipment", {}) or {}).get("cranes", [])
    crane = cranes[0] if cranes else {"crane_id": "C1"}

    relocations: List[Dict[str, Any]] = []
    crane_tasks: List[Dict[str, Any]] = []
    line_sequences: Dict[str, List[Dict[str, Any]]] = {lid: [] for lid in line_remaining.keys()}

    t_crane = 0.0
    seq_no: Dict[str, int] = {lid: 0 for lid in line_remaining.keys()}

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

    # 主循环：每一步选一个(line, plate)
    total_left = sum(len(v) for v in line_remaining.values())
    while total_left > 0:
        lid, pid = choose_next_action(plates, line_remaining, line_ready, t_crane, cfg, W)

        # relocations to satisfy LIFO
        blockers = list(plates[pid].get("blocked_by", []) or [])
        for b in blockers:
            if b in plates:
                do_relocate(b, lid, pid)

        # outbound action
        eta = max(t_crane, line_ready[lid])  # line_ready is when line can start cutting next
        start = eta
        finish = start + move_s
        t_crane = finish

        seq_no[lid] += 1
        line_sequences[lid].append({
            "plate_id": pid,
            "seq_no": seq_no[lid],
            "eta_seconds": eta,
            "start_seconds": start,
            "finish_seconds": finish
        })
        crane_tasks.append({
            "task_id": f"OUT_{lid}_{pid}_{seq_no[lid]}",
            "type": "outbound",
            "plate_id": pid,
            "crane_id": crane.get("crane_id", "C1"),
            "start_seconds": start,
            "finish_seconds": finish,
        })

        plates[pid]["status"] = "to_cut"
        # line_ready advances by cutting time once plate arrives
        line_ready[lid] = max(line_ready[lid], finish) + cut_s

        # remove from remaining
        line_remaining[lid] = [x for x in line_remaining[lid] if x != pid]
        total_left = sum(len(v) for v in line_remaining.values())

    plan = {
        "meta": {"run_id": "TO_BE_FILLED", "instance_id": instance["meta"]["instance_id"], "solver": "rolling_horizon", "created_at": now_iso()},
        "line_sequences": [{"cutting_line_id": lid, "sequence": seq} for lid, seq in line_sequences.items()],
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
    solver = "rolling_horizon"
    exp = cfg.get("experiment", "exp")
    run_id = gen_run_id(scenario, solver, exp)

    inst_path = Path(args.instance)
    if not inst_path.is_absolute():
        inst_path = ROOT / inst_path
    if not inst_path.exists():
        raise FileNotFoundError(f"Instance not found: {inst_path}")

    instance = read_json(inst_path)
    validate_json(instance, load_schema(ROOT / "schemas" / "instance.schema.json"), label="instance")

    plan = solve_rh(instance, cfg)
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
