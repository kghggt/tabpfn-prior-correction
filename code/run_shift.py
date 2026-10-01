"""自然先验偏移（子群漂移）场景。

与诱导设定的区别：**不重采样任何标签**。训练侧与测试侧是数据中天然存在的两个亚群，
二者的稀有类患病率不同，因此先验偏移来自数据本身而不是实验者的下采样。
这是回应"结论只在人为诱导的偏移上成立"这一质疑的关键实验。

用法（在 venues/prl 目录下）：
  python code/run_shift.py --datasets adult bank-marketing --versions v2 v2_5
  python code/run_shift.py --probe    # 只报告每份数据找到的子群与偏移量，不跑模型
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback

# 必须在导入 huggingface_hub / tabpfn 之前设置
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HOME", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".cache", "hf"))

import numpy as np  # noqa: E402

import config  # noqa: E402
import data  # noqa: E402
from runner import UNITS_DIR, log, run_split, unit_filename  # noqa: E402

SHIFT_CANDIDATES = config.SHIFT_DATASETS


MIN_SHIFT_RATIO = 1.25  # 测试侧患病率 / 训练侧患病率 的最低要求


def probe(names) -> list[dict]:
    """报告每份数据能否构造出有意义的天然偏移。"""
    rows = []
    for name in names:
        try:
            X_df, y, meta = data.load_openml_raw(name)
            picked = data.pick_subgroup_pair(X_df, y)
        except Exception as exc:  # noqa: BLE001
            log(f"{name}: 探测失败 {type(exc).__name__}: {str(exc)[:80]}")
            continue
        if picked is None:
            log(f"{name}: 无可用子群对")
            continue
        col, tr_vals, te_vals, p_tr, p_te = picked
        ratio = (p_te / p_tr) if p_tr > 0 else float("inf")
        rec = {
            "dataset": name,
            "openml_id": meta.get("openml_id"),
            "column": col,
            "train_values": tr_vals,
            "test_values": te_vals,
            "prev_train_min": p_tr,
            "prev_test_min": p_te,
            "shift_ratio": ratio,
            "n_total": int(len(y)),
        }
        rows.append(rec)
        flag = "KEEP" if ratio >= MIN_SHIFT_RATIO else "skip"
        log(
            f"{name:32s} col={col:18s} 训练侧{min(tr_vals)!s:>14s}->{p_tr:.3f} "
            f"测试侧{min(te_vals)!s:>14s}->{p_te:.3f} ratio={ratio:.2f} [{flag}]"
        )
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    ap.add_argument("--versions", nargs="*", default=config.MODEL_VERSIONS)
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--min-ratio", type=float, default=MIN_SHIFT_RATIO)
    args = ap.parse_args()

    names = args.datasets or SHIFT_CANDIDATES
    rows = probe(names)

    out_json = os.path.join(config.RESULTS_DIR, "shift_probe.json")
    with open(out_json, "w", encoding="utf-8") as fh:
        import json

        json.dump(rows, fh, indent=1)
    log(f"探测结果写入 {out_json}")

    if args.probe:
        return 0

    keep = [r["dataset"] for r in rows if r["shift_ratio"] >= args.min_ratio]
    log(f"保留 {len(keep)} 份数据（天然偏移比 >= {args.min_ratio}）: {keep}")

    n = 0
    total = len(keep) * len(args.seeds) * len(args.versions)
    for name in keep:
        for seed in args.seeds:
            try:
                split = data.build_subgroup_split(name, seed)
            except Exception as exc:  # noqa: BLE001
                log(f"{name} seed{seed} 切分失败: {exc!r}")
                log(traceback.format_exc())
                continue
            setting = split["setting"]
            for version in args.versions:
                n += 1
                out_path = os.path.join(
                    UNITS_DIR, unit_filename(name, seed, setting, version, "shift")
                )
                try:
                    payload = run_split(split, version, seed, out_path, grid="shift")
                    if payload is None:
                        log(f"[{n}/{total}] {name} {setting} seed{seed} {version} — 已存在，跳过")
                        continue
                    mt = payload["methods"]
                    log(
                        f"[{n}/{total}] {name} {setting} seed{seed} {version} — "
                        f"π_tr={payload['prior_train'][payload['minority_class']]:.3f} "
                        f"π_te={payload['prior_test'][payload['minority_class']]:.3f} | "
                        f"tabpfn {mt['tabpfn']['macro_f1']:.3f} / picl {mt['picl']['macro_f1']:.3f} / "
                        f"distpfn {mt['distpfn']['macro_f1']:.3f} ({payload['runtime_sec']}s)"
                    )
                except Exception as exc:  # noqa: BLE001
                    log(f"[{n}/{total}] {name} {setting} seed{seed} {version} — 失败: {exc!r}")
                    log(traceback.format_exc())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
