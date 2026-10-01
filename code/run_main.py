"""命令行入口。用法（均在 venues/prl 目录下执行）：

  python code/run_main.py --suite smoke
  python code/run_main.py --suite main12
  python code/run_main.py --suite rare12
  python code/run_main.py --suite extended --versions v2 v2_5
  python code/run_main.py --suite main12 --versions v3_5
"""

from __future__ import annotations

import argparse
import os
import sys

# 必须在导入 huggingface_hub / tabpfn 之前设置：
#   * huggingface.co 在本机不可达，走 hf-mirror.com 镜像
#   * HF 缓存重定向到工作区内（默认缓存目录在沙箱中不可写）
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HOME", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".cache", "hf"))
# 6GB 显存下减少碎片，降低 TabPFN 前向 OOM 的概率
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TABPFN_CHUNK", "256")

import config
from runner import log, run_grid

SUITES = {
    "smoke": dict(datasets=["diabetes", "wdbc"], seeds=[0], settings=["ind05"], versions=["v2"]),
    "main12": dict(
        datasets=config.MAIN12, seeds=config.SEEDS_MAIN, settings=config.SETTINGS, versions=config.MODEL_VERSIONS
    ),
    "rare12": dict(
        datasets=config.RARE12, seeds=config.SEEDS_RARE, settings=["natural"], versions=config.MODEL_VERSIONS
    ),
    "extended": dict(
        datasets=config.EXTENDED, seeds=config.SEEDS_EXTENDED, settings=["ind05"], versions=config.MODEL_VERSIONS
    ),
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", required=True, choices=list(SUITES))
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--seeds", nargs="*", type=int, default=None)
    ap.add_argument("--settings", nargs="*", default=None)
    ap.add_argument("--versions", nargs="*", default=None)
    args = ap.parse_args()

    plan = dict(SUITES[args.suite])
    if args.datasets:
        plan["datasets"] = args.datasets
    if args.seeds:
        plan["seeds"] = args.seeds
    if args.settings:
        plan["settings"] = args.settings
    if args.versions:
        plan["versions"] = args.versions

    log(f"命令行: {' '.join(sys.argv)}")
    run_grid(
        datasets=plan["datasets"],
        seeds=plan["seeds"],
        settings=plan["settings"],
        versions=plan["versions"],
        tag=f"[suite={args.suite}]",
        # 带 RUN_TAG 的消融/留出运行用自己的 grid 名，避免和主网格混在一起聚合
        grid=(config.RUN_TAG or args.suite),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
