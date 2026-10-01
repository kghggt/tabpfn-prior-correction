#!/usr/bin/env bash
# 补强实验总入口：单进程串行，避免 6GB 显存上的争抢。
# 用法（venues/prl 目录）：bash run_strengthen.sh
set -u
cd "$(dirname "$0")"
PY="C:/Users/15983/AppData/Local/conda/conda/envs/csac_tabpfn/python.exe"
OUT="logs/strengthen_$(date +%H%M%S).txt"

{
  echo "启动 $(date)"

  echo "=== 1 主网格受显存预算影响的三张表重跑 ==="
  "$PY" -u code/run_main.py --suite main12 --datasets phoneme wilt spambase --seeds 0 1 2

  echo "=== 2 rare12 自然稀有面板 ==="
  "$PY" -u code/run_main.py --suite rare12 --seeds 0 1 2

  echo "=== 3 extended 扩展网格 ==="
  "$PY" -u code/run_main.py --suite extended

  echo "=== 4 天然子群偏移 ==="
  "$PY" -u code/run_shift.py --seeds 0 1 2

  echo "=== 5 留出表（未参与协议选择）==="
  TABPFN_RUN_TAG=heldout "$PY" -u code/run_main.py --suite main12 \
      --datasets pc3 climate-model --settings ind05 --seeds 0 1 2

  echo "=== 6 estimator 数量消融 ==="
  for n in 8 16 32; do
    TABPFN_N_EST=$n TABPFN_RUN_TAG=est$n "$PY" -u code/run_main.py \
        --suite main12 --settings ind05 --seeds 0 --versions v2
  done

  echo "=== 7 可部署诊断量回填 ==="
  "$PY" -u code/run_diag.py

  echo "=== 8 聚合与出表 ==="
  "$PY" -u code/analyze.py
  "$PY" -u code/export_tables.py

  echo "完成 $(date)"
} > "$OUT" 2>&1
echo "日志: $OUT"
