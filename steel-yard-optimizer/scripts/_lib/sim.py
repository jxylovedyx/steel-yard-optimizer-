from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from typing import Any, Dict, List, Tuple


def _get_move_seconds(instance: Dict[str, Any], cfg: Dict[str, Any]) -> float:
    cranes = (instance.get("equipment", {}) or {}).get("cranes", [])
    if not cranes:
        return 60.0
    c = cranes[0]
    pick = float(c.get("pick_seconds", 25))
    place = float(c.get("place_seconds", 25))
    sec_per_m = float(c.get("seconds_per_meter", (instance.get("equipment", {}) or {}).get("travel_model", {}).get("seconds_per_meter", 0.8)))
    travel_m = float(((cfg.get("solver_params", {}) or {}).get("travel_meters_per_move", 20)))
    return pick + place + sec_per_m * travel_m


def simulate_plan(instance: Dict[str, Any], plan: Dict[str, Any], cfg: Dict[str, Any]) -> Dict[str, float]:
    """
    KPI:
    - makespan_seconds: 完成所有线供料动作+切割完成的时间
    - relocation_count: 翻板/移库次数（plan.relocations长度）
    - line_starvation_seconds: 各切割线因缺料导致的空转时间总和（基于切割节拍）
    - crane_utilization: 起重机忙碌时间 / makespan
    """
    move_s = _get_move_seconds(instance, cfg)
    cut_s = float(((cfg.get("solver_params", {}) or {}).get("cut_seconds_per_plate", 45)))

    
    demand_map: Dict[str, List[str]] = {}
    for d in instance.get("demands", []) or []:
        demand_map[str(d["cutting_line_id"])] = [str(x) for x in (d.get("items", []) or [])]

    seq_map: Dict[str, List[Dict[str, Any]]] = {}
    for ls in plan.get("line_sequences", []) or []:
        seq_map[str(ls["cutting_line_id"])] = list(ls.get("sequence", []) or [])

    
    line_ready: Dict[str, float] = {lid: 0.0 for lid in demand_map.keys()}
    starvation_total = 0.0

    
    for lid, seq in seq_map.items():
        t = 0.0
        for step in seq:
            arrive = float(step.get("finish_seconds", 0.0))
            start_cut = max(arrive, t)
            if start_cut > t:
                starvation_total += (start_cut - t)
            t = start_cut + cut_s
        line_ready[lid] = t

    makespan = max(line_ready.values()) if line_ready else 0.0

    
    reloc_cnt = int(len(plan.get("relocations", []) or []))
    out_cnt = 0
    for lid, seq in seq_map.items():
        out_cnt += len(seq)

    crane_busy = (reloc_cnt + out_cnt) * move_s
    util = float(crane_busy / makespan) if makespan > 1e-9 else 0.0
    util = min(1.0, max(0.0, util))

    return {
        "makespan_seconds": float(round(makespan, 3)),
        "relocation_count": int(reloc_cnt),
        "line_starvation_seconds": float(round(starvation_total, 3)),
        "crane_utilization": float(round(util, 6)),
    }
