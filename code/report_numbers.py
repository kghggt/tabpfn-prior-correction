"""把 results/summary.json 压成一段可直接抄进稿件的数字清单。

用途：正文里每一个数字都要能在产物里追到，这个脚本把追的过程自动化，
避免手抄。只写 stdout 与 results/report_numbers.md。
"""

from __future__ import annotations

import json
import os
from collections import defaultdict

import numpy as np

import config

LBL = {
    "tabpfn": "TabPFN",
    "os": "+OS",
    "smote": "+SMOTE",
    "balctx": "BalCtx",
    "thr": "Thr",
    "picl": "PICL",
    "distpfn": "DistPFN",
    "distpfn_t": "DistPFN-T",
    "picl_full": "PICL-Full",
    "picl_oracle": "Oracle",
    "logreg": "LogReg",
    "xgboost_w": "XGB-w",
    "catboost_w": "CatB-w",
    "lightgbm_w": "LGBM-w",
}


def fmt_p(p):
    if p is None or p != p:
        return "--"
    if p < 1e-3:
        return "<1e-3"
    return f"{p:.3f}"


def main() -> int:
    path = os.path.join(config.RESULTS_DIR, "summary.json")
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)

    out = []
    out.append(f"# 数字清单（{data['n_units']} 个单元）\n")

    for key, s in sorted(data["grids"].items()):
        grid, setting, version = key.split("|")
        out.append(f"\n## {grid} | {setting} | {version}  (n={s['n_datasets']})")
        means = sorted(s["methods"].items(), key=lambda kv: -kv[1]["mean"])
        out.append("  " + "  ".join(f"{LBL.get(k, k)}={v['mean']:.3f}" for k, v in means))
        for tk, t in s["tests"].items():
            a, b = tk.split("_vs_")
            out.append(
                f"  {LBL.get(a,a)} vs {LBL.get(b,b)}: {t['mean_a']:.3f} / {t['mean_b']:.3f} "
                f"Δ={t['mean_diff']:+.3f} wins={t['wins_a']}/{t['n']} p={fmt_p(t['p_one_sided'])}"
            )
        for tk, t in s.get("tests_cluster", {}).items():
            a, b = tk.split("_vs_")
            out.append(
                f"  [簇内合并] {LBL.get(a,a)} vs {LBL.get(b,b)}: Δ={t['mean_diff']:+.3f} "
                f"wins={t['wins_a']}/{t['n']} p={fmt_p(t['p_one_sided'])}"
            )

    # 崩塌分层
    out.append("\n## 崩塌分层 (ratio = 预测少数类率 / 测试集少数类率)")
    for key, rows in sorted(data.get("collapse", {}).items()):
        if not rows:
            continue
        buckets = [(0.0, 0.25), (0.25, 0.75), (0.75, 1.30), (1.30, 1e9)]
        names = ["collapsed", "partial", "agreed", "overshoot"]
        parts = []
        for (lo, hi), nm in zip(buckets, names):
            sel = [r for r in rows if r["ratio"] == r["ratio"] and lo <= r["ratio"] < hi]
            if not sel:
                continue
            parts.append(
                f"{nm}: n={len(sel)} ΔPICL={np.mean([r['d_picl'] for r in sel]):+.3f} "
                f"ΔDistPFN={np.mean([r['d_distpfn'] for r in sel]):+.3f}"
            )
        out.append(f"  {key}: " + " | ".join(parts))

    # 可部署判据
    out.append("\n## 可部署判据（训练留出片保真度）")
    for key, rows in sorted(data.get("detector", {}).items()):
        if not rows:
            continue
        buckets = [(0.0, 0.34), (0.34, 0.67), (0.67, 1.0), (1.0, 1e9)]
        names = ["<0.34", "0.34-0.67", "0.67-1.0", ">=1.0"]
        parts = []
        for (lo, hi), nm in zip(buckets, names):
            sel = [r for r in rows if lo <= r["fidelity"] < hi]
            if not sel:
                continue
            parts.append(
                f"{nm}: n={len(sel)} ΔPICL={np.mean([r['d_picl'] for r in sel]):+.3f} "
                f"ΔDistPFN={np.mean([r['d_distpfn'] for r in sel]):+.3f}"
            )
        out.append(f"  {key}: " + " | ".join(parts))

    text = "\n".join(out)
    print(text)
    dst = os.path.join(config.RESULTS_DIR, "report_numbers.md")
    with open(dst, "w", encoding="utf-8") as fh:
        fh.write(text + "\n")
    print(f"\n写入 {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
