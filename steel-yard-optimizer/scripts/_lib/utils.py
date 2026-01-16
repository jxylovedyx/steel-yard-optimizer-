from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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


def gen_run_id(scenario: object, solver: str, exp: str) -> str:
    """Short, filesystem-safe run id.
    scenario can be str/dict/anything; dict will be hashed and truncated.
    """
    import hashlib
    from datetime import datetime, timezone
    import re as _re

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    if isinstance(scenario, str):
        scen = scenario
    elif isinstance(scenario, dict):
        name = str(scenario.get("name", "scenario"))
        h = hashlib.sha1(repr(sorted(scenario.items())).encode("utf-8")).hexdigest()[:8]
        scen = f"{name}-{h}"
    else:
        scen = str(scenario)

    def safe(s: str) -> str:
        s = s.strip().replace(" ", "_")
        s = _re.sub(r"[^A-Za-z0-9._-]+", "_", s)
        return s[:60]

    return f"{ts}_{safe(scen)}_{safe(solver)}_{safe(exp)}"
