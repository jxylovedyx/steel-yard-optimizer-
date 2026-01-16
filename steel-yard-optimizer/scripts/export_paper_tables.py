from __future__ import annotations

import argparse
import ast
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


TABLE_I = [
    {"Notation": "N", "Meaning": "The number of cutting plates and operation periods."},
    {"Notation": "S", "Meaning": "The number of stacks."},
    {"Notation": "L", "Meaning": "The number of cutting lines."},
    {"Notation": "ES_s", "Meaning": "The number of empty slots in stack s."},
    {"Notation": "B_i", "Meaning": "The number of blocking plates above cutting plate i."},
    {"Notation": "B_i^m", "Meaning": "The number of adjacent blocking plates above i."},
    {"Notation": "P_is", "Meaning": "Equal 1 if cutting plate i is in stack s, otherwise 0."},
    {"Notation": "DT", "Meaning": "Delivery time from the yard to cutting lines."},
    {"Notation": "CT_i", "Meaning": "Cutting hour (time) of plate i."},
    {"Notation": "K_ij", "Meaning": "Equal 1 if i,j are cut in the same line, otherwise 0."},
    {"Notation": "O_i", "Meaning": "Set of cutting plates located above cutting plate i."},
    {"Notation": "RT_i", "Meaning": "Retrieval time of cutting plate i."},
    {"Notation": "MT_ss'", "Meaning": "Relocation time from stack s to s'."},
    {"Notation": "OT_i", "Meaning": "Minimum outbound consumption of cutting plate i."},
    {"Notation": "N_lowest", "Meaning": "Set of plates in the lowest slot of each stack."},
]

TABLE_II = [
    {"Variable": "x_in", "Type": "Binary", "Meaning": "Equal 1 if plate i leaves the yard in period n, otherwise 0."},
    {"Variable": "y_sn", "Type": "Integer", "Meaning": "The number of plates relocated from stack s in period n."},
    {"Variable": "z_sn", "Type": "Integer", "Meaning": "The number of plates received by stack s in period n."},
    {"Variable": "pt_n", "Type": "Continuous", "Meaning": "Operation time of period n."},
    {"Variable": "ot_i", "Type": "Continuous", "Meaning": "Outbound time of plate i."},
    {"Variable": "r_ij", "Type": "Binary", "Meaning": "Equal 1 if i and j are cut in same line and i is before j, otherwise 0."},
    {"Variable": "ct_i", "Type": "Continuous", "Meaning": "Cutting completion time of plate i."},
    {"Variable": "c_max", "Type": "Continuous", "Meaning": "Makespan."},
]


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def read_json(p: Path) -> Any:
    return json.loads(p.read_text(encoding="utf-8"))


@dataclass
class TimeModel:
    DT: float = 120.0
    CT_default: float = 60.0
    RT_default: float = 50.0
    MT_default: float = 60.0


def load_time_model(run_dir: Path) -> TimeModel:
    tm = TimeModel()
    snap = run_dir / "config_snapshot.yaml"
    if snap.exists():
        try:
            import yaml  # type: ignore
            cfg = yaml.safe_load(snap.read_text(encoding="utf-8")) or {}
            t = cfg.get("time_model", {}) if isinstance(cfg, dict) else {}
            tm.DT = float(t.get("delivery_seconds", t.get("DT", tm.DT)))
            tm.CT_default = float(t.get("cut_seconds_per_plate", tm.CT_default))
            tm.RT_default = float(t.get("pick_seconds", 25.0)) + float(t.get("place_seconds", 25.0))
            tm.MT_default = float(t.get("relocation_seconds", t.get("MT", tm.MT_default)))
        except Exception:
            pass
    return tm


def stack_key(area: str, stack_id: str) -> str:
    area = (area or "").strip()
    stack_id = (stack_id or "").strip()
    if area:
        return f"{area}:{stack_id}"
    return stack_id


def parse_loc(x: Any) -> Dict[str, Any]:
    
    if x is None:
        return {}
    s = str(x).strip()
    if not s:
        return {}
    try:
        obj = ast.literal_eval(s)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


_TASK_RE = re.compile(r"^R_(?P<line>L\d+)_(?P<target>P\d+)_(?P<moved>P\d+)_(?P<k>\d+)$")


def parse_target_plate(task_id: str) -> Optional[str]:
    m = _TASK_RE.match(task_id.strip())
    if not m:
        return None
    return m.group("target")


def read_exports(run_dir: Path) -> Dict[str, pd.DataFrame]:
    exp_dir = run_dir / "exports"
    out: Dict[str, pd.DataFrame] = {}
    for name in ["line_sequence.csv", "relocations.csv", "crane_tasks.csv"]:
        p = exp_dir / name
        if p.exists():
            out[name] = pd.read_csv(p)
    return out


def compute_notations(instance: Dict[str, Any], tm: TimeModel) -> Dict[str, Any]:
    plates = instance.get("plates", [])
    stacks = instance.get("stacks", instance.get("yard_stacks", []))
    demands = instance.get("demands", [])
    line_ids = sorted({str(d.get("cutting_line_id")) for d in demands if d.get("cutting_line_id")})

    # stack capacity
    max_levels_map: Dict[str, int] = {}
    for s in stacks:
        sk = stack_key(str(s.get("area", "")), str(s.get("stack_id", s.get("id", ""))))
        ml = int(s.get("max_levels", s.get("capacity", 16) or 16))
        max_levels_map[sk] = ml

    # plate->stack, level
    plate_pos: Dict[str, Tuple[str, int]] = {}
    stack_levels: Dict[str, List[Tuple[int, str]]] = defaultdict(list)
    for p in plates:
        pid = str(p.get("plate_id"))
        sk = stack_key(str(p.get("area", "")), str(p.get("stack_id", "")))
        lv = int(p.get("level", 1))
        plate_pos[pid] = (sk, lv)
        stack_levels[sk].append((lv, pid))
    for sk in stack_levels:
        stack_levels[sk].sort(key=lambda t: t[0])

    ES_s = {}
    for sk, ml in max_levels_map.items():
        ES_s[sk] = max(0, ml - len(stack_levels.get(sk, [])))

    O_i, B_i, B_i_m, P_is_stack = {}, {}, {}, {}
    for pid, (sk, lv) in plate_pos.items():
        above = [pp for (lvl, pp) in stack_levels[sk] if lvl > lv]
        O_i[pid] = above
        B_i[pid] = len(above)
        B_i_m[pid] = min(len(above), 5)
        P_is_stack[pid] = sk

    N_lowest = []
    for sk, items in stack_levels.items():
        if items:
            N_lowest.append(items[0][1])

    return {
        "N": len(plates),
        "S": len(max_levels_map),
        "L": len(line_ids),
        "lines": line_ids,
        "ES_s": ES_s,
        "B_i": B_i,
        "B_i^m": B_i_m,
        "P_is_stack": P_is_stack,
        "O_i": O_i,
        "RT_i_default": tm.RT_default,
        "DT": tm.DT,
        "CT_i_default": tm.CT_default,
        "MT_default": tm.MT_default,
        "N_lowest": N_lowest,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance", required=True)
    ap.add_argument("--run-dir", required=True)   
    ap.add_argument("--out", required=True)       
    args = ap.parse_args()

    inst_path = Path(args.instance)
    run_dir = Path(args.run_dir)
    out_dir = Path(args.out)
    ensure_dir(out_dir)

    # write Table I/II
    pd.DataFrame(TABLE_I).to_csv(out_dir / "table_I_notations.csv", index=False)
    pd.DataFrame(TABLE_II).to_csv(out_dir / "table_II_variables.csv", index=False)

    instance = read_json(inst_path)
    tm = load_time_model(run_dir)
    notations = compute_notations(instance, tm)
    (out_dir / "notations_values.json").write_text(json.dumps(notations, ensure_ascii=False, indent=2), encoding="utf-8")

    exp = read_exports(run_dir)
    if "line_sequence.csv" not in exp:
        raise FileNotFoundError(f"Missing exports/line_sequence.csv in {run_dir}")

    line_seq = exp["line_sequence.csv"].copy()
    # required cols exist in your data: cutting_line_id, seq_no, plate_id, eta_seconds, start_seconds, finish_seconds
    for c in ["cutting_line_id", "seq_no", "plate_id"]:
        if c not in line_seq.columns:
            raise ValueError(f"line_sequence.csv missing column: {c}")

   
    if "eta_seconds" in line_seq.columns:
        line_seq["_k"] = line_seq["eta_seconds"].astype(float)
    elif "start_seconds" in line_seq.columns:
        line_seq["_k"] = line_seq["start_seconds"].astype(float)
    else:
        line_seq["_k"] = line_seq["seq_no"].astype(float)

    line_seq["cutting_line_id"] = line_seq["cutting_line_id"].astype(str)
    line_seq["plate_id"] = line_seq["plate_id"].astype(str)

    global_order = line_seq.sort_values(["_k", "cutting_line_id", "seq_no"]).reset_index(drop=True)
    plate_to_n = {pid: i + 1 for i, pid in enumerate(global_order["plate_id"].tolist())}


    x_in_df = pd.DataFrame([{"i": pid, "n": plate_to_n[pid], "x_in": 1} for pid in global_order["plate_id"]])
 
    if "finish_seconds" in line_seq.columns:
        ct_df = line_seq[["plate_id", "finish_seconds", "cutting_line_id"]].copy()
        ct_df = ct_df.rename(columns={"plate_id": "i", "finish_seconds": "ct_i", "cutting_line_id": "line"})
        ct_df["ct_i"] = ct_df["ct_i"].astype(float)
        cmax = float(ct_df["ct_i"].max()) if len(ct_df) else 0.0
    else:
        ct_df = pd.DataFrame(columns=["i", "line", "ct_i"])
        cmax = 0.0
    cmax_df = pd.DataFrame([{"c_max": cmax}])

   
    DT = float(notations.get("DT", tm.DT))
    if "eta_seconds" in line_seq.columns:
        base = line_seq[["plate_id", "eta_seconds"]].copy()
        base["ot_i"] = base["eta_seconds"].astype(float) - DT
    elif "start_seconds" in line_seq.columns:
        base = line_seq[["plate_id", "start_seconds"]].copy()
        base["ot_i"] = base["start_seconds"].astype(float) - DT
    else:
        base = line_seq[["plate_id"]].copy()
        base["ot_i"] = 0.0
    base["ot_i"] = base["ot_i"].clip(lower=0.0)
    ot_map = {str(r["plate_id"]): float(r["ot_i"]) for _, r in base.iterrows()}
    ot_df = pd.DataFrame([{"i": pid, "ot_i": ot_map.get(pid, 0.0)} for pid in global_order["plate_id"]])

    # pt_n from ot_i differences in global period order
    prev = 0.0
    pt_rows = []
    for pid in global_order["plate_id"]:
        n = plate_to_n[pid]
        ot = float(ot_map.get(pid, 0.0))
        pt = max(0.0, ot - prev)
        pt_rows.append({"n": n, "pt_n": pt, "plate_i": pid})
        prev = ot
    pt_df = pd.DataFrame(pt_rows)

    # r_ij within each line by seq_no
    rij_rows = []
    for lid, grp in line_seq.sort_values(["cutting_line_id", "seq_no"]).groupby("cutting_line_id"):
        seq = grp["plate_id"].astype(str).tolist()
        for a in range(len(seq)):
            for b in range(a + 1, len(seq)):
                rij_rows.append({"i": seq[a], "j": seq[b], "r_ij": 1, "line": str(lid)})
    rij_df = pd.DataFrame(rij_rows)

    # y_sn / z_sn from relocations.csv (map to period by target plate in task_id)
    y_sn = defaultdict(int)
    z_sn = defaultdict(int)
    reloc = exp.get("relocations.csv")
    if reloc is not None and len(reloc) > 0:
        # expected cols in your data: task_id, plate_id, from, to, reason, estimated_seconds
        for _, rr in reloc.iterrows():
            tid = str(rr.get("task_id", ""))
            target = parse_target_plate(tid)
            if target is None:
                continue
            n = int(plate_to_n.get(target, 1))

            frm = parse_loc(rr.get("from"))
            to = parse_loc(rr.get("to"))

            if "stack_id" in frm:
                s_from = stack_key(str(frm.get("area", "")), str(frm.get("stack_id", "")))
                y_sn[(s_from, n)] += 1
            elif "buffer_id" in frm:
                y_sn[(f"BUF:{frm.get('buffer_id')}", n)] += 1

            if "stack_id" in to:
                s_to = stack_key(str(to.get("area", "")), str(to.get("stack_id", "")))
                z_sn[(s_to, n)] += 1
            elif "buffer_id" in to:
                z_sn[(f"BUF:{to.get('buffer_id')}", n)] += 1

    y_df = pd.DataFrame([{"s": s, "n": n, "y_sn": v} for (s, n), v in y_sn.items()])
    z_df = pd.DataFrame([{"s": s, "n": n, "z_sn": v} for (s, n), v in z_sn.items()])

    vv = out_dir / "variables_values"
    ensure_dir(vv)
    x_in_df.to_csv(vv / "x_in.csv", index=False)
    y_df.to_csv(vv / "y_sn.csv", index=False)
    z_df.to_csv(vv / "z_sn.csv", index=False)
    pt_df.to_csv(vv / "pt_n.csv", index=False)
    ot_df.to_csv(vv / "ot_i.csv", index=False)
    rij_df.to_csv(vv / "r_ij.csv", index=False)
    ct_df.to_csv(vv / "ct_i.csv", index=False)
    cmax_df.to_csv(vv / "c_max.csv", index=False)

    print(f"[OK] paper tables -> {out_dir.resolve()}")
    print(f"[OK] variables -> {(vv).resolve()}")


if __name__ == "__main__":
    main()
