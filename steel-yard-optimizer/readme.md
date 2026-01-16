# steel-yard-optimizer

面向多条切割线的钢板堆场出库（outbound）与翻板（relocation）优化：  
在满足 **LIFO/层序约束** 的前提下，输出可执行的作业计划，并最小化：
1) 切割作业总完工时间（makespan）  
2) 翻板/移库次数（relocation_count）

本项目支持：
- 数据标准化：raw -> interim -> instance.json
- 求解器对比：baseline / heuristic / rolling-horizon（RH）
- 学习排序（可选）：XGBoost Ranker 学习启发式决策并用于推理
- 结果导出：plan.json / metrics.json / exports/*.csv

---

## 1. 问题定义

### 输入（Input）
1) 堆场存储状态（库存）
- 每块钢板 plate_id、规格（厚宽长重）、项目/分段（可选）
- 堆叠位置：area、stack_id、level
- LIFO 约束：blocked_by（上层阻挡板列表）

2) 切割线需求
- cutting_line_id -> items(plate_id list)

3) 设备与参数
- 吊机：pick/place 时间、移动速度（seconds_per_meter）、载重（可扩展多吊车）
- 切割节拍：cut_seconds_per_plate
- 缓冲位：buffer（可扩展容量约束）

### 输出（Output）
作业计划（plan.json）：
- line_sequences：每条切割线的供料顺序与时间戳（eta/start/finish）
- relocations：为满足 LIFO 发生的翻板/移库动作
- crane_tasks：统一的吊机任务时间线（relocate/outbound）

评估结果（metrics.json）：
- makespan_seconds：总完工时间
- relocation_count：翻板次数
- line_starvation_seconds：缺料导致的空转时间（近似）
- crane_utilization：吊机利用率（近似）

---

## 2. 项目结构

configs/ # 参数配置
data/
raw/ # 原始数据（csv）
interim/ # 标准化后的中间数据（csv）
instances/ # 实例（instance.json）
results/<RUN_ID>/ # 输出（plan/metrics/exports）
models/ # 学习模型（xgb_ranker.json）
schemas/ # JSON schema
scripts/
_lib/ # 工具库
import_raw_data.py # raw -> interim
make_instance.py # interim/toy -> instance.json
run_solver.py # baseline rule solver
run_solver_heuristic.py# heuristic expert
run_solver_rh.py # rolling horizon solver（双目标近似）
compute_metrics.py # plan -> metrics
export_plan.py # plan -> csv exports
generate_synthetic_raw.py
infer_and_export.sh
learn/
make_ltr_dataset.py
train_xgb_ranker.py
run_solver_xgb.py # 推理：用ranker给排序+仍遵守LIFO


---

## 3. 安装依赖

建议在 conda env `steel` 中：
```bash
pip install -r requirements.txt

4. 快速开始：端到端跑通（推荐先用合成数据）
4.1 生成较大 synthetic raw（自动扩容避免 capacity 报错）
python3 scripts/generate_synthetic_raw.py \
  --out data/raw \
  --areas 4 \
  --stacks-per-area 120 \
  --max-levels 16 \
  --plates 10000 \
  --lines 8 \
  --demand-per-line 1200 \
  --seed 7 \
  --auto-capacity
4.2 raw -> interim（标准化）
python3 scripts/import_raw_data.py --raw-dir data/raw --out data/interim
4.3 interim -> instance.json
python3 scripts/make_instance.py \
  --mode interim \
  --interim-dir data/interim \
  --out "data/instances/plantA_$(date +%Y%m%d).json"
4.4 运行求解（baseline / heuristic / rh）
INST="data/instances/plantA_$(date +%Y%m%d).json"

./scripts/infer_and_export.sh "$INST" baseline
./scripts/infer_and_export.sh "$INST" heuristic
./scripts/infer_and_export.sh "$INST" rh

输出在：data/results/<RUN_ID>/
  plan.json
  metrics.json
  exports/
    line_sequence.csv
    crane_tasks.csv
    relocations.csv

5. 学习模型
5.1 先生成“专家轨迹”
INST="data/instances/plantA_$(date +%Y%m%d).json"
for i in $(seq 1 50); do
  python3 scripts/run_solver_heuristic.py --config configs/baseline.yaml --instance "$INST"
done
5.2 生成 LTR 数据集并训练 XGBoost Ranker（GPU）
python3 scripts/learn/make_ltr_dataset.py --max-runs 500 --max-candidates 128

python3 scripts/learn/train_xgb_ranker.py \
  --data data/datasets/ltr.parquet \
  --groups data/datasets/ltr.groups.txt \
  --out models/xgb_ranker.json \
  --ndcg-k 10 \
  --rounds 8000 \
  --early-stop 300 \
  --max-bin 256 \
  --max-leaves 256 \
  --lr 0.05
5.3 推理（ranker 排序 + LIFO 翻板）
python3 scripts/run_solver_xgb.py \
  --instance "$INST" \
  --model models/xgb_ranker.json \
  --scenario prod \
  --exp xgb

RUN_ID="$(ls -1dt data/results/* | head -n 1 | xargs -n1 basename)"
python3 scripts/compute_metrics.py --plan "data/results/$RUN_ID/plan.json"
python3 scripts/export_plan.py --plan "data/results/$RUN_ID/plan.json" --out "data/results/$RUN_ID/exports"
cat "data/results/$RUN_ID/metrics.json"
