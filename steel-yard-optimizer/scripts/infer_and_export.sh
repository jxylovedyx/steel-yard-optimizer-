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

# ---- export paper tables (Table I/II + derived variables) ----
if [ -f "$INSTANCE" ] && [ -d "data/results/$RUN_ID" ]; then
  mkdir -p artifacts/paper
  python3 scripts/export_paper_tables.py \
    --instance "$INSTANCE" \
    --run-dir "data/results/$RUN_ID" \
    --out "artifacts/paper/$RUN_ID" || true
fi
