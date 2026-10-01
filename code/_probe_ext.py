"""探测扩展数据集：能否解析、是否二分类、规模是否可用。

只做只读探测。为避免下载超大表，先读取 NumberOfInstances 元数据过滤。
每处理一个数据集立即 flush 并增量落盘，避免进程被杀时丢失全部结果。

用法：
  python -u code/_probe_ext.py
"""

from __future__ import annotations

import json
import os
import sys
import traceback
import warnings

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402

import config  # noqa: E402
import data  # noqa: E402

warnings.filterwarnings("ignore")

MAX_ROWS = 12000
MIN_ROWS = 300
MIN_MINORITY_PCT = 0.02

OUT_PATH = os.path.join(config.RESULTS_DIR, "ext_probe.json")


def main() -> int:
    import openml

    index = data._build_index()
    rows = []
    fh = open(OUT_PATH, "w", encoding="utf-8", buffering=1)

    for name in config.EXTENDED:
        rec = {"name": name, "status": "?"}
        try:
            did = data.resolve_dataset_id(name, index)
            rec["openml_id"] = did
            if did is None:
                rec["status"] = "unresolved"
                print(f"{name:45s} UNRESOLVED", flush=True)
                rows.append(rec)
                continue
            # 先取元数据，避免下载超大表
            meta_only = openml.datasets.get_dataset(did, download_data=False)
            try:
                n_rows = int(meta_only.qualities.get("NumberOfInstances", -1))
            except Exception:  # noqa: BLE001
                n_rows = -1
            rec["meta_rows"] = n_rows
            if n_rows > 0 and n_rows > MAX_ROWS * 3:
                rec["status"] = "too_large_meta"
                print(f"{name:45s} too_large_meta n={n_rows}", flush=True)
                rows.append(rec)
                continue

            X, y, meta = data.load_openml_binary(name)
            n = len(y)
            p_min = float(np.bincount(y).min() / n)
            rec.update({"n_rows": n, "n_features": int(X.shape[1]), "minority_pct": p_min})
            if n < MIN_ROWS:
                rec["status"] = "too_small"
            elif n > MAX_ROWS:
                rec["status"] = "too_large"
            elif p_min < MIN_MINORITY_PCT:
                rec["status"] = "too_rare"
            else:
                rec["status"] = "ok"
            print(
                f"{name:45s} {rec['status']:14s} id={did:<7d} n={n:<7d} "
                f"d={rec.get('n_features', 0):<5d} min%={p_min:.3f}",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001
            rec["status"] = f"error:{type(exc).__name__}"
            print(f"{name:45s} ERROR {type(exc).__name__}: {str(exc)[:70]}", flush=True)
        rows.append(rec)
        fh.write(json.dumps(rows, indent=1))
        fh.write("\n")
        fh.truncate()
        fh.seek(0)

    fh.close()
    ok = [r["name"] for r in rows if r["status"] == "ok"]
    print(f"\n可用: {len(ok)} / {len(config.EXTENDED)}", flush=True)
    print(json.dumps(ok, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        raise SystemExit(1)
