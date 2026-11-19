#!/usr/bin/env bash
# run.sh — 批量运行 47 个 case 并生成金标准 CSV
#
# 依赖：仅 Python 3.9+ 标准库（无需 numpy / pandas / requests）
# 原脚本 trajectory_v3_pure.py 保持不动，本流程只使用新增文件。

set -euo pipefail

PYTHON="${PYTHON:-python3}"
DATASET="${1:-万象渊鉴·场域诊断测试集.txt}"

echo "== 1) 单 case / 整份测试集的超细轨迹与每轮计分 =="
# 模式 A：ultra_trace.csv（内部验证，超细步长）
# 模式 B：ultra_turn_scores.csv（交付给 nik13 的每轮计分，每轮仅一行）
"${PYTHON}" trajectory_v3_ultra.py \
  --input "${DATASET}" \
  --char_step 10 \
  --output_traj ultra_trace.csv \
  --output_turn ultra_turn_scores.csv

echo
echo "== 2) 批量汇总 47 个 case =="
# --input_dir 支持目录，也支持单个测试集文件
"${PYTHON}" aggregator.py \
  --input_dir "${DATASET}" \
  --char_step 10 \
  --output_gold gold_standard_47_cases.csv \
  --output_diag diagnosis_summary.csv

echo
echo "== 完成 =="
echo "交付：gold_standard_47_cases.csv"
echo "诊断：diagnosis_summary.csv"
echo "内部：ultra_trace.csv / ultra_turn_scores.csv"

# 单 case 示例（JSON 含 messages 数组时）：
#   "${PYTHON}" trajectory_v3_ultra.py --input case_v2_01.json --char_step 10 \
#       --output_traj traj.csv --output_turn turn.csv
