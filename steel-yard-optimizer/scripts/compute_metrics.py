from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
from pathlib import Path
from typing import Any, Dict, Optional

from scripts._lib.config import resolve_config
from scripts._lib.schema import load_schema, validate_json
from scripts._lib.sim import simulate_plan
from scripts._lib.utils import now_iso, read_json, write_json

ROOT = Path(__file__).resolve().parents[1]


def infer_instance_path(plan_path: Path) -> Optional[Path]:
  
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
