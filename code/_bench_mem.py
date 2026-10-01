"""显存/耗时基准：在 6GB 卡上找 TabPFN 单次前向的安全工作点。

只写 logs/ 与 results/bench_mem.json，不跑任何论文实验。
用法：python -u code/_bench_mem.py
"""

from __future__ import annotations

import json
import os
import sys
import time

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import config  # noqa: E402

GRID = [
    (4096, 1000),
    (4096, 512),
    (4096, 256),
    (2048, 1000),
    (2048, 512),
    (1024, 1000),
    (1024, 512),
]
FEATURES = [20, 57]
VERSIONS = ["v2"]


def one(version: str, n_tr: int, n_te: int, d: int) -> dict:
    from tabpfn import TabPFNClassifier

    rng = np.random.default_rng(0)
    X_tr = rng.normal(size=(n_tr, d)).astype(np.float32)
    y_tr = (rng.random(n_tr) < 0.05).astype(int)
    if y_tr.sum() < 2:
        y_tr[:2] = 1
    X_te = rng.normal(size=(n_te, d)).astype(np.float32)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    rec = {"version": version, "n_train": n_tr, "n_test": n_te, "d": d}
    try:
        clf = TabPFNClassifier(
            model_path=config.model_path(version),
            n_estimators=config.N_ESTIMATORS,
            ignore_pretraining_limits=True,
            random_state=0,
            device="cuda",
        )
        clf.fit(X_tr, y_tr)
        P = np.asarray(clf.predict_proba(X_te), dtype=float)
        rec.update(
            ok=True,
            sec=round(time.time() - t0, 1),
            peak_gb=round(torch.cuda.max_memory_allocated() / 1024**3, 2),
            pshape=list(P.shape),
        )
    except Exception as exc:  # noqa: BLE001
        rec.update(ok=False, err=f"{type(exc).__name__}: {str(exc)[:90]}")
    try:
        del clf
    except Exception:  # noqa: BLE001
        pass
    torch.cuda.empty_cache()
    return rec


def main() -> int:
    out = []
    for ver in VERSIONS:
        for d in FEATURES:
            for n_tr, n_te in GRID:
                rec = one(ver, n_tr, n_te, d)
                out.append(rec)
                print(json.dumps(rec, ensure_ascii=False), flush=True)
                if not rec["ok"]:
                    # OOM 会污染 CUDA 上下文，后续测量不可信；换下一个维度前先清干净
                    torch.cuda.empty_cache()
                    time.sleep(2)
    path = os.path.join(config.RESULTS_DIR, "bench_mem.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    print(f"写入 {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
