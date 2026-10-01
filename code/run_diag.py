"""回填可部署诊断量 holdout_diag。

主网格最初的一批单元生成时尚未记录该字段，本脚本只为这些单元补算：
在每个 (dataset, seed, setting, version) 上重建切分，对训练侧留出片做一次前向，
记录"模型在自己分布上预测少数类的比例 / 该片的真实少数类比例"。

该量只用训练数据，不碰任何测试标签，因此是部署时可得的。
用法：python code/run_diag.py [--only-main12]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
import traceback
from datetime import datetime

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HOME", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".cache", "hf"))

import numpy as np  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

import config  # noqa: E402
from data import build_split  # noqa: E402
from runner import UNITS_DIR, get_tabpfn, log  # noqa: E402


def compute_diag(split, version, seed):
    X_tr, y_tr = split["X_tr"], split["y_tr"]
    minority = split["minority_class"]
    ho_idx, fit_idx = train_test_split(
        np.arange(len(y_tr)),
        test_size=config.HOLDOUT_FRACTION,
        random_state=seed,
        stratify=y_tr,
    )
    clf = get_tabpfn(version, seed)
    clf.fit(X_tr[fit_idx], y_tr[fit_idx])
    P_ho = np.asarray(clf.predict_proba(X_tr[ho_idx]), dtype=float)
    pred_rate = float(np.mean((P_ho[:, minority] >= 0.5).astype(int)))
    true_rate = float(np.mean(y_tr[ho_idx] == minority))
    return {
        "true_minority_rate": true_rate,
        "predicted_minority_rate": pred_rate,
        "fidelity": (pred_rate / true_rate) if true_rate > 0 else None,
        "n_holdout": int(len(ho_idx)),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(UNITS_DIR, "*.json")))
    todo = []
    for p in paths:
        with open(p, encoding="utf-8") as fh:
            u = json.load(fh)
        if u.get("holdout_diag") is not None:
            continue
        if u.get("grid", "main12") != "main12":
            continue
        todo.append((p, u))

    if args.limit:
        todo = todo[: args.limit]
    log(f"待补算单元: {len(todo)}")
    t_start = time.time()
    for i, (p, u) in enumerate(todo, 1):
        try:
            split = build_split(u["dataset"], u["seed"], u["setting"])
            diag = compute_diag(split, u["model_version"], u["seed"])
            u["holdout_diag"] = diag
            with open(p, "w", encoding="utf-8") as fh:
                json.dump(u, fh, indent=1)
            f = diag["fidelity"]
            fs = "nan" if f is None else f"{f:.2f}"
            log(f"[{i}/{len(todo)}] {u['dataset']} {u['setting']} seed{u['seed']} {u['model_version']} fidelity={fs}")
        except Exception as exc:  # noqa: BLE001
            log(f"[{i}/{len(todo)}] {u['dataset']} {u['setting']} seed{u['seed']} {u['model_version']} 失败: {exc!r}")
            log(traceback.format_exc())
    log(f"补算结束，用时 {time.time() - t_start:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
