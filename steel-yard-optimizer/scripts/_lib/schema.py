from __future__ import annotations

import sys
from pathlib import Path

# ensure project root on sys.path (so `import scripts._lib...` works)
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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
