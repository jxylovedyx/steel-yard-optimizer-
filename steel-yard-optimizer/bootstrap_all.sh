#!/usr/bin/env bash
set -euo pipefail

ROOT="$(pwd)"

mkdir -p scripts/_lib scripts/learn schemas configs data/{raw,interim,instances,results} models

###############################################################################
# requirements.txt
###############################################################################
cat > requirements.txt <<'REQ'
numpy>=1.26
pandas>=2.0
pyyaml>=6.0
jsonschema>=4.20
pyarrow>=14.0
xgboost>=3.1.0
scikit-learn>=1.4
joblib>=1.3
ortools>=9.8
REQ

###############################################################################
# schemas
###############################################################################
cat > schemas/instance.schema.json <<'JSON'
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "title": "Steel Yard Instance",
  "type": "object",
  "required": ["meta", "yard", "equipment", "plates", "demands"],
  "properties": {
    "meta": {
      "type": "object",
      "required": ["instance_id", "scenario", "created_at"],
      "properties": {
        "instance_id": {"type": "string"},
        "scenario": {"type": "string"},
        "created_at": {"type": "string"}
      },
      "additionalProperties": true
    },
    "yard": {
      "type": "object",
      "required": ["stacks"],
      "properties": {
        "stacks": {
          "type": "array",
          "items": {
            "type": "object",
            "required": ["area", "stack_id", "max_levels"],
            "properties": {
              "area": {"type": "string"},
              "stack_id": {"type": "string"},
              "max_levels": {"type": "integer", "minimum": 1},
              "x": {"type": "number"},
              "y": {"type": "number"}
            },
            "additionalProperties": true
          }
        }
      },
      "additionalProperties": true
    },
    "equipment": {
      "type": "object",
      "required": ["cranes"],
      "properties": {
        "cranes": {
          "type": "array",
          "minItems": 1,
          "items": {
            "type": "object",
            "required": ["crane_id", "max_load_ton", "pick_seconds", "place_seconds"],
            "properties": {
              "crane_id": {"type": "string"},
              "max_load_ton": {"type": "number"},
              "pick_seconds": {"type": "number"},
              "place_seconds": {"type": "number"},
              "seconds_per_meter": {"type": "number"}
            },
            "additionalProperties": true
          }
        },
        "travel_model": {
          "type": "object",
          "properties": {
            "seconds_per_meter": {"type": "number"}
          },
          "additionalProperties": true
        }
      },
      "additionalProperties": true
    },
    "plates": {
      "type": "array",
      "items": {
        "type": "object",
        "required": ["plate_id", "spec", "location", "blocked_by", "status", "eligible_lines"],
        "properties": {
          "plate_id": {"type": "string"},
          "spec": {
            "type": "object",
            "required": ["thickness_mm", "width_mm", "length_mm", "weight_ton"],
            "properties": {
              "thickness_mm": {"type": "number"},
              "width_mm": {"type": "number"},
              "length_mm": {"type": "number"},
              "weight_ton": {"type": "number"}
            },
            "additionalProperties": true
          },
          "location": {
            "type": "object",
            "required": ["area", "stack_id", "level"],
            "properties": {
              "area": {"type": "string"},
              "stack_id": {"type": "string"},
              "level": {"type": "integer", "minimum": 1}
            },
            "additionalProperties": true
          },
          "blocked_by": {
            "type": "array",
            "items": {"type": "string"}
          },
          "status": {"type": "string"},
          "eligible_lines": {
            "type": "array",
            "items": {"type": "string"}
          }
        },
        "additionalProperties": true
      }
    },
    "demands": {
      "type": "array",
      "items": {
        "type": "object",
        "required": ["cutting_line_id", "items"],
        "properties": {
          "cutting_line_id": {"type": "string"},
          "items": {
            "type": "array",
            "items": {"type": "string"}
          }
        },
        "additionalProperties": true
      }
    }
  },
  "additionalProperties": true
}
JSON

cat > schemas/plan.schema.json <<'JSON'
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "title": "Steel Yard Plan",
  "type": "object",
  "required": ["meta", "line_sequences", "relocations", "crane_tasks", "warnings"],
  "properties": {
    "meta": {
      "type": "object",
      "required": ["run_id", "instance_id", "solver", "created_at"],
      "properties": {
        "run_id": {"type": "string"},
        "instance_id": {"type": "string"},
        "solver": {"type": "string"},
        "created_at": {"type": "string"}
      },
      "additionalProperties": true
    },
    "line_sequences": {
      "type": "array",
      "items": {
        "type": "object",
        "required": ["cutting_line_id", "sequence"],
        "properties": {
          "cutting_line_id": {"type": "string"},
          "sequence": {
            "type": "array",
            "items": {
              "type": "object",
              "required": ["plate_id", "seq_no", "eta_seconds", "start_seconds", "finish_seconds"],
              "properties": {
                "plate_id": {"type": "string"},
                "seq_no": {"type": "integer", "minimum": 1},
                "eta_seconds": {"type": "number"},
                "start_seconds": {"type": "number"},
                "finish_seconds": {"type": "number"}
              },
              "additionalProperties": true
            }
          }
        },
        "additionalProperties": true
      }
    },
    "relocations": {
      "type": "array",
      "items": {"type": "object"},
      "additionalProperties": true
    },
    "crane_tasks": {
      "type": "array",
      "items": {"type": "object"},
      "additionalProperties": true
    },
    "warnings": {
      "type": "array",
      "items": {"type": "string"}
    }
  },
  "additionalProperties": true
}
JSON

cat > schemas/metrics.schema.json <<'JSON'
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "title": "Steel Yard Metrics",
  "type": "object",
  "required": ["meta", "kpi"],
  "properties": {
    "meta": {
      "type": "object",
      "required": ["run_id", "instance_id", "created_at"],
      "properties": {
        "run_id": {"type": "string"},
        "instance_id": {"type": "string"},
        "created_at": {"type": "string"}
      },
      "additionalProperties": true
    },
    "kpi": {
      "type": "object",
      "required": ["makespan_seconds", "relocation_count", "line_starvation_seconds", "crane_utilization"],
      "properties": {
        "makespan_seconds": {"type": "number"},
        "relocation_count": {"type": "integer"},
        "line_starvation_seconds": {"type": "number"},
        "crane_utilization": {"type": "number"}
      },
      "additionalProperties": true
    }
  },
  "additionalProperties": true
}
JSON

###############################################################################
# configs/baseline.yaml (create if missing)
###############################################################################
if [ ! -f configs/baseline.yaml ]; then
cat > configs/baseline.yaml <<'YAML'
scenario: toy
experiment: baseline
io:
  output_dir: data/results

solver_params:
  travel_meters_per_move: 20
  cut_seconds_per_plate: 45
  buffer_ids: ["BUF1"]

# rolling horizon (RH) params
rh:
  window_per_line: 60
  choose_per_step: 1
  eps_makespan: 0.01
YAML
fi

###############################################################################
# python package markers
###############################################################################
cat > scripts/__init__.py <<'PY'
# scripts package
PY

cat > scripts/_lib/__init__.py <<'PY'
# internal libs
PY

###############################################################################
# scripts/_lib/utils.py
###############################################################################
cat > scripts/_lib/utils.py <<'PY'
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Dict

import pandas as pd


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def read_json(p: Path) -> Any:
    return json.loads(p.read_text(encoding="utf-8"))


def write_json(p: Path, obj: Any) -> None:
    ensure_dir(p.parent)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def read_csv(p: Path) -> pd.DataFrame:
    return pd.read_csv(p)


def write_csv(p: Path, df: pd.DataFrame) -> None:
    ensure_dir(p.parent)
    df.to_csv(p, index=False)


def dump_yaml(p: Path, cfg: Dict[str, Any]) -> None:
    import yaml
    ensure_dir(p.parent)
    p.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def gen_run_id(scenario: str, solver: str, exp: str) -> str:
    ts = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    return f"{ts}_{scenario}_{solver}_{exp}"
PY

###############################################################################
# scripts/_lib/config.py
###############################################################################
cat > scripts/_lib/config.py <<'PY'
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


def load_yaml(p: Path) -> Dict[str, Any]:
    return yaml.safe_load(p.read_text(encoding="utf-8")) or {}


def deep_merge(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(a)
    for k, v in b.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def resolve_config(cfg_path: str) -> Dict[str, Any]:
    p = Path(cfg_path)
    if not p.is_absolute():
        p = Path(__file__).resolve().parents[2] / p
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {p}")
    return load_yaml(p)
PY

###############################################################################
# scripts/_lib/schema.py
###############################################################################
cat > scripts/_lib/schema.py <<'PY'
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from jsonschema import Draft202012Validator


def load_schema(p: Path) -> Dict[str, Any]:
    import json
    return json.loads(p.read_text(encoding="utf-8"))


def validate_json(obj: Any, schema: Dict[str, Any], label: str = "json") -> None:
    v = Draft202012Validator(schema)
    errors = sorted(v.iter_errors(obj), key=lambda e: e.path)
    if errors:
        lines = [f"[{label}] schema validation failed with {len(errors)} errors:"]
        for e in errors[:30]:
            path = ".".join([str(x) for x in e.path])
            lines.append(f" - path='{path}': {e.message}")
        raise ValueError("\n".join(lines))
PY

###############################################################################
# scripts/_lib/sim.py  (核心：仿真评估 KPI)
###############################################################################
cat > scripts/_lib/sim.py <<'PY'
from __future__ import annotations

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

    # 建 demand map：line -> ordered items
    demand_map: Dict[str, List[str]] = {}
    for d in instance.get("demands", []) or []:
        demand_map[str(d["cutting_line_id"])] = [str(x) for x in (d.get("items", []) or [])]

    # plan sequences
    seq_map: Dict[str, List[Dict[str, Any]]] = {}
    for ls in plan.get("line_sequences", []) or []:
        seq_map[str(ls["cutting_line_id"])] = list(ls.get("sequence", []) or [])

    # 每条线：切割开始时间取决于“供料动作完成时间”和“上一块切完时间”
    line_ready: Dict[str, float] = {lid: 0.0 for lid in demand_map.keys()}
    starvation_total = 0.0

    # 我们按 plan 中每条线 sequence 给出的 finish_seconds 作为“板到线”的时间
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

    # crane busy time: outbound moves + relocation moves
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
PY

###############################################################################
# scripts/import_raw_data.py  (raw -> interim 标准化)
###############################################################################
cat > scripts/import_raw_data.py <<'PY'
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import pandas as pd

from scripts._lib.utils import ensure_dir, read_csv, write_csv


REQUIRED_RAW = ["plates.csv", "yard_stacks.csv", "equipment_cranes.csv", "demands.csv"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default="data/raw")
    ap.add_argument("--out", default="data/interim")
    args = ap.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out)
    ensure_dir(out_dir)

    missing = [str(raw_dir / f) for f in REQUIRED_RAW if not (raw_dir / f).exists()]
    if missing:
        raise FileNotFoundError(f"Missing raw files: {missing}")

    # plates
    plates = read_csv(raw_dir / "plates.csv")
    # normalize eligible_lines (pipe separated -> list-like string kept; instance builder will split)
    if "eligible_lines" not in plates.columns:
        plates["eligible_lines"] = ""
    if "status" not in plates.columns:
        plates["status"] = "in_yard"

    # yard stacks
    yard = read_csv(raw_dir / "yard_stacks.csv")
    if "max_levels" not in yard.columns:
        yard["max_levels"] = 8

    cranes = read_csv(raw_dir / "equipment_cranes.csv")
    if "seconds_per_meter" not in cranes.columns:
        cranes["seconds_per_meter"] = 0.8

    demands = read_csv(raw_dir / "demands.csv")
    # demands expected columns: cutting_line_id, plate_id, seq_no
    if not {"cutting_line_id", "plate_id"}.issubset(set(demands.columns)):
        raise ValueError("demands.csv must have cutting_line_id, plate_id columns")
    if "seq_no" not in demands.columns:
        demands["seq_no"] = demands.groupby("cutting_line_id").cumcount() + 1

    # write interim
    write_csv(out_dir / "plates.csv", plates)
    write_csv(out_dir / "yard_stacks.csv", yard)
    write_csv(out_dir / "equipment_cranes.csv", cranes)
    write_csv(out_dir / "demands.csv", demands)

    print(f"[OK] interim generated at: {out_dir.resolve()}")
    for f in REQUIRED_RAW:
        p = out_dir / f
        print(f" - {p} ({p.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
PY

###############################################################################
# scripts/make_instance.py  (toy / interim -> instance.json)
###############################################################################
cat > scripts/make_instance.py <<'PY'
from __future__ import annotations

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
    if need > len(all_ids):
        raise ValueError(f"Need {need} plates for demands but only {len(all_ids)} exist")
    demands = []
    cur = 0
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
    ap.add_argument("--plates", type=int, default=2000)
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
PY

###############################################################################
# scripts/run_solver.py (baseline rule: 按需求顺序 + 必要翻板)
###############################################################################
cat > scripts/run_solver.py <<'PY'
from __future__ import annotations

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
PY

###############################################################################
# scripts/run_solver_heuristic.py (可学习专家：更偏向减少翻板+兼顾效率)
###############################################################################
cat > scripts/run_solver_heuristic.py <<'PY'
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple

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


def plate_cost(p: Dict[str, Any]) -> float:
    # 代价函数（可解释）：被压多/层深 => 更不想先取；顶部优先；重量略惩罚
    loc = p.get("location", {}) or {}
    spec = p.get("spec", {}) or {}
    blocked_cnt = float(len(p.get("blocked_by", []) or []))
    level = float(loc.get("level", 0.0))
    weight = float(spec.get("weight_ton", 0.0))
    is_top = 1.0 if blocked_cnt == 0 else 0.0
    return blocked_cnt * 100.0 + level * 1.0 + weight * 0.1 - is_top * 2.0


def heuristic_solve(instance: Dict[str, Any], cfg: Dict[str, Any]) -> Dict[str, Any]:
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
        remaining = [str(x) for x in (d.get("items", []) or []) if str(x) in plates]
        seq_out: List[Dict[str, Any]] = []
        t_line = t_line_finish.get(lid, 0.0)
        step = 0

        while remaining:
            remaining.sort(key=lambda pid: plate_cost(plates[pid]))
            pid = remaining.pop(0)
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

    return {
        "meta": {"run_id": "TO_BE_FILLED", "instance_id": instance["meta"]["instance_id"], "solver": "heuristic_expert", "created_at": now_iso()},
        "line_sequences": line_sequences,
        "relocations": relocations,
        "crane_tasks": crane_tasks,
        "warnings": [],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/baseline.yaml")
    ap.add_argument("--instance", required=True)
    args = ap.parse_args()

    cfg = resolve_config(args.config)
    scenario = cfg.get("scenario", "toy")
    solver = "heuristic_expert"
    exp = cfg.get("experiment", "exp")
    run_id = gen_run_id(scenario, solver, exp)

    inst_path = Path(args.instance)
    if not inst_path.is_absolute():
        inst_path = ROOT / inst_path
    if not inst_path.exists():
        raise FileNotFoundError(f"Instance not found: {inst_path}")

    instance = read_json(inst_path)
    validate_json(instance, load_schema(ROOT / "schemas" / "instance.schema.json"), label="instance")

    plan = heuristic_solve(instance, cfg)
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
PY

###############################################################################
# scripts/run_solver_rh.py (滚动窗口：双目标(词典序) + 约束(LIFO) + 共享起重机近似)
###############################################################################
cat > scripts/run_solver_rh.py <<'PY'
from __future__ import annotations

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
PY

###############################################################################
# scripts/compute_metrics.py (plan + instance -> metrics.json)
###############################################################################
cat > scripts/compute_metrics.py <<'PY'
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Optional

from scripts._lib.config import resolve_config
from scripts._lib.schema import load_schema, validate_json
from scripts._lib.sim import simulate_plan
from scripts._lib.utils import now_iso, read_json, write_json

ROOT = Path(__file__).resolve().parents[1]


def infer_instance_path(plan_path: Path) -> Optional[Path]:
    # try sibling input_manifest.json
    manifest = plan_path.parent / "input_manifest.json"
    if manifest.exists():
        m = read_json(manifest)
        p = Path(str(m.get("instance_path", "")))
        if not p.is_absolute():
            p = ROOT / p
        return p if p.exists() else None
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True)
    ap.add_argument("--instance", default="")
    ap.add_argument("--config", default="configs/baseline.yaml")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    cfg = resolve_config(args.config)

    plan_path = Path(args.plan)
    if not plan_path.is_absolute():
        plan_path = ROOT / plan_path
    if not plan_path.exists():
        raise FileNotFoundError(plan_path)

    inst_path = Path(args.instance) if args.instance else (infer_instance_path(plan_path) or Path(""))
    if not str(inst_path):
        raise FileNotFoundError("instance path not provided and cannot infer from input_manifest.json")
    if not inst_path.is_absolute():
        inst_path = ROOT / inst_path
    if not inst_path.exists():
        raise FileNotFoundError(inst_path)

    plan = read_json(plan_path)
    instance = read_json(inst_path)

    validate_json(instance, load_schema(ROOT / "schemas" / "instance.schema.json"), label="instance")
    validate_json(plan, load_schema(ROOT / "schemas" / "plan.schema.json"), label="plan")

    kpi = simulate_plan(instance, plan, cfg)
    metrics = {"meta": {"run_id": plan["meta"]["run_id"], "instance_id": instance["meta"]["instance_id"], "created_at": now_iso()}, "kpi": kpi}
    validate_json(metrics, load_schema(ROOT / "schemas" / "metrics.schema.json"), label="metrics")

    out_path = Path(args.out) if args.out else (plan_path.parent / "metrics.json")
    if not out_path.is_absolute():
        out_path = ROOT / out_path
    write_json(out_path, metrics)
    print(f"[OK] metrics -> {out_path}")


if __name__ == "__main__":
    main()
PY

###############################################################################
# scripts/export_plan.py (plan -> exports/*.csv)
###############################################################################
cat > scripts/export_plan.py <<'PY'
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from scripts._lib.utils import ensure_dir, read_json, write_csv

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True)
    ap.add_argument("--out", default="data/exports")
    args = ap.parse_args()

    plan_path = Path(args.plan)
    if not plan_path.is_absolute():
        plan_path = ROOT / plan_path
    if not plan_path.exists():
        raise FileNotFoundError(plan_path)

    out_dir = Path(args.out)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    ensure_dir(out_dir)

    plan = read_json(plan_path)

    # line sequence
    rows = []
    for ls in plan.get("line_sequences", []) or []:
        lid = str(ls["cutting_line_id"])
        for s in ls.get("sequence", []) or []:
            rows.append({
                "cutting_line_id": lid,
                "seq_no": int(s["seq_no"]),
                "plate_id": str(s["plate_id"]),
                "eta_seconds": float(s["eta_seconds"]),
                "start_seconds": float(s["start_seconds"]),
                "finish_seconds": float(s["finish_seconds"])
            })
    df_seq = pd.DataFrame(rows).sort_values(["cutting_line_id", "seq_no"]) if rows else pd.DataFrame(columns=["cutting_line_id","seq_no","plate_id","eta_seconds","start_seconds","finish_seconds"])
    write_csv(out_dir / "line_sequence.csv", df_seq)

    # crane tasks
    df_tasks = pd.DataFrame(plan.get("crane_tasks", []) or [])
    if not df_tasks.empty and "start_seconds" in df_tasks.columns:
        df_tasks = df_tasks.sort_values(["start_seconds"])
    write_csv(out_dir / "crane_tasks.csv", df_tasks if not df_tasks.empty else pd.DataFrame(columns=["task_id","type","plate_id","crane_id","start_seconds","finish_seconds"]))

    # relocations
    df_reloc = pd.DataFrame(plan.get("relocations", []) or [])
    write_csv(out_dir / "relocations.csv", df_reloc if not df_reloc.empty else pd.DataFrame(columns=["task_id","plate_id","reason","estimated_seconds"]))

    print(f"[OK] exports -> {out_dir.resolve()}")
    for f in ["line_sequence.csv", "crane_tasks.csv", "relocations.csv"]:
        p = out_dir / f
        print(f" - {p} ({p.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
PY

###############################################################################
# scripts/generate_synthetic_raw.py (更大数据、可自动扩容)
###############################################################################
cat > scripts/generate_synthetic_raw.py <<'PY'
from __future__ import annotations

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
PY

###############################################################################
# learn scripts: make_ltr_dataset.py (含shuffle防泄漏)
###############################################################################
cat > scripts/learn/make_ltr_dataset.py <<'PY'
from __future__ import annotations

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

                # IMPORTANT: avoid order leakage
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
PY

###############################################################################
# learn scripts: train_xgb_ranker.py (XGB 3.x + CUDA + QuantileDMatrix ref)
###############################################################################
cat > scripts/learn/train_xgb_ranker.py <<'PY'
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

FEATURES = [
    "thickness_mm","width_mm","length_mm","weight_ton","level","blocked_cnt","is_top",
    "stack_height","stack_count","area_hash","stack_hash",
    "step","demand_pos","remaining_pos","remaining_cnt",
]

def load_groups(p: Path) -> List[int]:
    txt = p.read_text(encoding="utf-8").strip()
    if not txt:
        return []
    return [int(x) for x in txt.splitlines() if x.strip()]

def split_by_groups(df: pd.DataFrame, groups: List[int], test_size: float, seed: int) -> Tuple[pd.DataFrame, List[int], pd.DataFrame, List[int]]:
    idx = np.arange(len(groups))
    tr_idx, va_idx = train_test_split(idx, test_size=test_size, random_state=seed, shuffle=True)
    tr_set = set(tr_idx.tolist())
    va_set = set(va_idx.tolist())

    start = 0
    tr_chunks = []
    va_chunks = []
    g_tr: List[int] = []
    g_va: List[int] = []
    for gi, gsz in enumerate(groups):
        end = start + gsz
        if gi in tr_set:
            tr_chunks.append(df.iloc[start:end])
            g_tr.append(gsz)
        elif gi in va_set:
            va_chunks.append(df.iloc[start:end])
            g_va.append(gsz)
        start = end

    df_tr = pd.concat(tr_chunks, axis=0) if tr_chunks else df.iloc[0:0].copy()
    df_va = pd.concat(va_chunks, axis=0) if va_chunks else df.iloc[0:0].copy()
    return df_tr, g_tr, df_va, g_va

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/datasets/ltr.parquet")
    ap.add_argument("--groups", default="data/datasets/ltr.groups.txt")
    ap.add_argument("--out", default="models/xgb_ranker.json")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--ndcg-k", type=int, default=10)
    ap.add_argument("--rounds", type=int, default=8000)
    ap.add_argument("--early-stop", type=int, default=300)
    ap.add_argument("--max-bin", type=int, default=256)
    ap.add_argument("--max-leaves", type=int, default=256)
    ap.add_argument("--lr", type=float, default=0.05)
    args = ap.parse_args()

    df = pd.read_parquet(ROOT / args.data)
    groups = load_groups(ROOT / args.groups)

    if df.empty or not groups or sum(groups) != len(df):
        raise RuntimeError("bad dataset or groups")

    for c in FEATURES + ["label"]:
        if c not in df.columns:
            raise ValueError(f"missing column: {c}")

    df[FEATURES] = df[FEATURES].astype(np.float32)
    df["label"] = df["label"].astype(np.float32)

    df_tr, g_tr, df_va, g_va = split_by_groups(df, groups, args.test_size, args.seed)

    X_tr = df_tr[FEATURES].to_numpy(dtype=np.float32, copy=False)
    y_tr = df_tr["label"].to_numpy(dtype=np.float32, copy=False)
    X_va = df_va[FEATURES].to_numpy(dtype=np.float32, copy=False)
    y_va = df_va["label"].to_numpy(dtype=np.float32, copy=False)

    dtr = xgb.QuantileDMatrix(X_tr, label=y_tr, feature_names=FEATURES, max_bin=args.max_bin)
    dva = xgb.QuantileDMatrix(X_va, label=y_va, feature_names=FEATURES, max_bin=args.max_bin, ref=dtr)
    dtr.set_group(g_tr)
    dva.set_group(g_va)

    params = {
        "device": "cuda",
        "tree_method": "hist",
        "objective": "rank:pairwise",
        "eval_metric": [f"ndcg@{args.ndcg_k}"],
        "grow_policy": "lossguide",
        "max_leaves": int(args.max_leaves),
        "min_child_weight": 1.0,
        "sampling_method": "gradient_based",
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "learning_rate": float(args.lr),
        "lambda": 2.0,
        "alpha": 0.0,
        "gamma": 0.0,
        "random_state": args.seed,
    }

    print("[INFO] xgboost:", xgb.__version__)
    print("[INFO] train rows:", len(df_tr), "groups:", len(g_tr), "avg_group:", (len(df_tr)/max(1,len(g_tr))))
    print("[INFO] val rows:", len(df_va), "groups:", len(g_va), "avg_group:", (len(df_va)/max(1,len(g_va))))
    print("[INFO] params:", {k: params[k] for k in ["device","tree_method","objective","grow_policy","max_leaves","sampling_method","learning_rate"]})

    booster = xgb.train(
        params=params,
        dtrain=dtr,
        num_boost_round=args.rounds,
        evals=[(dtr, "train"), (dva, "val")],
        early_stopping_rounds=args.early_stop,
        verbose_eval=50,
    )

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    booster.save_model(out_path)
    print(f"[OK] saved model -> {out_path}")
    print("[OK] best_iteration:", booster.best_iteration)
    print("[OK] best_score:", booster.best_score)

if __name__ == "__main__":
    main()
PY

###############################################################################
# scripts/run_solver_xgb.py (推理用：ranker给排序，仍遵守LIFO翻板)
###############################################################################
cat > scripts/run_solver_xgb.py <<'PY'
from __future__ import annotations

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
PY

###############################################################################
# scripts/infer_and_export.sh (交付用：一键推理/评估/导出)
###############################################################################
cat > scripts/infer_and_export.sh <<'SH'
#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

INSTANCE="${1:-data/instances/toy_instance.json}"
MODE="${2:-rh}"       # baseline | heuristic | rh | xgb
MODEL="${3:-models/xgb_ranker.json}"

echo "[INFO] instance=$INSTANCE mode=$MODE model=$MODEL"

if [ "$MODE" = "baseline" ]; then
  python3 scripts/run_solver.py --config configs/baseline.yaml --instance "$INSTANCE"
elif [ "$MODE" = "heuristic" ]; then
  python3 scripts/run_solver_heuristic.py --config configs/baseline.yaml --instance "$INSTANCE"
elif [ "$MODE" = "rh" ]; then
  python3 scripts/run_solver_rh.py --config configs/baseline.yaml --instance "$INSTANCE"
elif [ "$MODE" = "xgb" ]; then
  python3 scripts/run_solver_xgb.py --instance "$INSTANCE" --model "$MODEL" --scenario prod --exp xgb
else
  echo "[ERR] unknown mode: $MODE"
  exit 2
fi

RUN_ID="$(ls -1dt data/results/* | head -n 1 | xargs -n1 basename)"
echo "[OK] RUN_ID=$RUN_ID"

python3 scripts/compute_metrics.py --plan "data/results/$RUN_ID/plan.json"
python3 scripts/export_plan.py --plan "data/results/$RUN_ID/plan.json" --out "data/results/$RUN_ID/exports"

echo "========== KPI =========="
cat "data/results/$RUN_ID/metrics.json"
echo "========================="
SH
chmod +x scripts/infer_and_export.sh

echo "[OK] bootstrap done."
echo "Next:"
echo "  pip install -r requirements.txt"
echo "  python3 scripts/make_instance.py --mode toy --out data/instances/toy_instance.json"
echo "  ./scripts/infer_and_export.sh data/instances/toy_instance.json rh"
