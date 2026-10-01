"""聚合 results/units/*.json，做显著性检验，输出 results/summary.json。

统计口径与论文一致：先在数据集内对种子取均值，得到每个数据集一个数；
再在十二个（或 n 个）数据集上做单侧 Wilcoxon 符号秩检验（alternative='greater'）。
所有数字均由本脚本从运行产物算出，不存在手写值。
"""

from __future__ import annotations

import glob
import json
import os
from collections import defaultdict

import numpy as np
from scipy.stats import wilcoxon

import config

UNITS_DIR = os.path.join(config.RESULTS_DIR, "units")


def load_units() -> list[dict]:
    units = []
    for path in sorted(glob.glob(os.path.join(UNITS_DIR, "*.json"))):
        with open(path, encoding="utf-8") as fh:
            units.append(json.load(fh))
    return units


def per_dataset_means(units, setting, version, method, metric="macro_f1", grid=None):
    """返回 {dataset: 该数据集跨种子均值}。"""
    acc = defaultdict(list)
    for u in units:
        if u["setting"] != setting or u["model_version"] != version:
            continue
        if grid is not None and u.get("grid", "main12") != grid:
            continue
        if method not in u["methods"]:
            continue
        val = u["methods"][method].get(metric)
        if val is None or (isinstance(val, float) and np.isnan(val)):
            continue
        acc[u["dataset"]].append(float(val))
    return {k: float(np.mean(v)) for k, v in acc.items()}


def paired_test(a: dict, b: dict, alternative="greater"):
    """a、b 为 {dataset: value}；在共同数据集上做配对检验。

    返回 (mean_a, mean_b, mean_diff, n_common, wins_a, p_value)。
    """
    common = sorted(set(a) & set(b))
    if len(common) < 3:
        return None
    va = np.array([a[k] for k in common], dtype=float)
    vb = np.array([b[k] for k in common], dtype=float)
    diff = va - vb
    try:
        res = wilcoxon(diff, alternative=alternative, zero_method="wilcox", method="exact")
        p = float(res.pvalue)
    except Exception:  # noqa: BLE001
        p = float("nan")
    return {
        "mean_a": float(va.mean()),
        "mean_b": float(vb.mean()),
        "mean_diff": float(diff.mean()),
        "n": len(common),
        "wins_a": int(np.sum(diff > 1e-9)),
        "ties": int(np.sum(np.abs(diff) <= 1e-9)),
        "p_one_sided": p,
    }


def collapse_clusters(per_method: dict, clusters: dict) -> dict:
    """把同源簇内的数据集先平均成一个观测，缓解簇内相关性对检验的 inflated power。"""
    buckets: dict[str, list] = {}
    for ds, v in per_method.items():
        buckets.setdefault(clusters.get(ds, ds), []).append(v)
    return {k: float(np.mean(v)) for k, v in buckets.items()}


def summarize(units, datasets, setting, version, metric="macro_f1", grid=None):
    """返回该 (grid, setting, version) 下的方法汇总与一组配对检验。"""
    in_scope = [
        u
        for u in units
        if u["setting"] == setting
        and u["model_version"] == version
        and u["dataset"] in datasets
        and (grid is None or u.get("grid", "main12") == grid)
    ]
    method_names = sorted({m for u in in_scope for m in u["methods"]})
    per_method = {
        m: per_dataset_means(in_scope, setting, version, m, metric, grid=grid) for m in method_names
    }

    summary = {
        "setting": setting,
        "model_version": version,
        "metric": metric,
        "n_datasets": len({u["dataset"] for u in in_scope}),
        "methods": {
            m: {"mean": float(np.mean(list(v.values()))), "n": len(v), "per_dataset": v}
            for m, v in per_method.items()
            if v
        },
        "tests": {},
    }

    pairs = [
        ("picl", "tabpfn"),
        ("picl", "distpfn"),
        ("picl", "distpfn_t"),
        ("distpfn", "tabpfn"),
        ("distpfn_t", "tabpfn"),
        ("picl", "thr"),
        ("picl", "os"),
        ("picl", "balctx"),
        ("picl", "smote"),
        ("distpfn", "os"),
        ("picl", "picl_oracle"),
        ("picl", "logreg"),
        ("picl", "xgboost_w"),
    ]
    for a, b in pairs:
        if a not in per_method or b not in per_method:
            continue
        r = paired_test(per_method[a], per_method[b])
        if r:
            summary["tests"][f"{a}_vs_{b}"] = r

    # rare panel：同源簇内先平均，再做一次检验，作为对簇内相关性的稳健性检查
    if grid == "rare12" and config.RARE_CLUSTERS:
        cl_a = {m: collapse_clusters(v, config.RARE_CLUSTERS) for m, v in per_method.items()}
        summary["tests_cluster"] = {}
        for a, b in pairs:
            if a not in cl_a or b not in cl_a:
                continue
            r = paired_test(cl_a[a], cl_a[b])
            if r:
                summary["tests_cluster"][f"{a}_vs_{b}"] = r
    return summary


def collapse_diagnostics(units, setting, version, grid=None):
    """崩塌强度分层：按 (预测少数类率 / 测试集少数类率) 分组看方法差值。"""
    rows = []
    for u in units:
        if u["setting"] != setting or u["model_version"] != version:
            continue
        if grid is not None and u.get("grid", "main12") != grid:
            continue
        mt = u["methods"]
        if "tabpfn" not in mt:
            continue
        minority = u["minority_class"]
        test_rate = float(u["prior_test"][minority])
        pred_rate = mt["tabpfn"]["predicted_minority_rate"]
        ratio = pred_rate / test_rate if test_rate > 0 else float("nan")
        rows.append(
            {
                "dataset": u["dataset"],
                "seed": u["seed"],
                "ratio": ratio,
                "test_rate": test_rate,
                "pred_rate": pred_rate,
                "d_picl": mt["picl"]["macro_f1"] - mt["tabpfn"]["macro_f1"],
                "d_distpfn": mt["distpfn"]["macro_f1"] - mt["tabpfn"]["macro_f1"],
                "d_distpfn_t": mt["distpfn_t"]["macro_f1"] - mt["tabpfn"]["macro_f1"],
            }
        )
    return rows


def detector_diagnostics(units, setting, version, grid="main12"):
    """可部署诊断：训练留出片的先验保真度 vs 校正带来的 macro-F1 变化。

    fidelity = 留出片上模型预测少数类的比例 / 该片真实少数类比例。
    只用训练数据，不需要任何测试标签，因此部署时可得。
    """
    rows = []
    for u in units:
        if u["setting"] != setting or u["model_version"] != version:
            continue
        if u.get("grid", "main12") != grid:
            continue
        diag = u.get("holdout_diag")
        mt = u["methods"]
        if not diag or "tabpfn" not in mt or "picl" not in mt:
            continue
        f = diag.get("fidelity")
        if f is None:
            continue
        rows.append(
            {
                "dataset": u["dataset"],
                "seed": u["seed"],
                "fidelity": float(f),
                "d_picl": mt["picl"]["macro_f1"] - mt["tabpfn"]["macro_f1"],
                "d_distpfn": mt["distpfn"]["macro_f1"] - mt["tabpfn"]["macro_f1"],
                "d_thr": mt["thr"]["macro_f1"] - mt["tabpfn"]["macro_f1"],
            }
        )
    return rows


def main() -> int:
    units = load_units()
    if not units:
        print("没有运行产物，先跑实验。")
        return 1

    out = {"n_units": len(units), "grids": {}, "collapse": {}, "detector": {}}
    grids = [
        ("main12", config.MAIN12, config.SETTINGS),
        ("rare12", config.RARE12, ["natural"]),
        ("extended", config.EXTENDED, ["ind05", "natural"]),
    ]
    # 自然偏移网格：setting 由子群列名决定，且每张表的列名不同，
    # 因此按 (dataset, setting) 动态收集，否则每张表都只看得到自己那一列，
    # n_datasets 恒为 1 而被门槛挡掉。
    shift_units = [u for u in units if u.get("grid") == "shift"]
    if shift_units:
        pairs = sorted({(u["dataset"], u["setting"]) for u in shift_units})
        for ds, st in pairs:
            grids.append(("shift", [ds], [st]))

    # 消融变体网格（RUN_TAG 产生的 grid）自动收集，避免每加一个消融就改脚本
    fixed = {g[0] for g in grids}
    for gname in sorted({u.get("grid", "main12") for u in units} - fixed):
        ds = sorted({u["dataset"] for u in units if u.get("grid", "main12") == gname})
        st = sorted({u["setting"] for u in units if u.get("grid", "main12") == gname})
        grids.append((gname, ds, st))

    for grid_name, datasets, settings in grids:
        present = sorted({u["dataset"] for u in units} & set(datasets))
        if not present:
            continue
        for setting in settings:
            for version in sorted({u["model_version"] for u in units}):
                s = summarize(units, present, setting, version, grid=grid_name)
                # 单表场景（自然偏移里每张表一个子群列）也要出，门槛只用于多表网格
                min_n = 1 if grid_name == "shift" else 3
                if s["n_datasets"] < min_n:
                    continue
                out["grids"][f"{grid_name}|{setting}|{version}"] = s
                out["collapse"][f"{grid_name}|{setting}|{version}"] = collapse_diagnostics(
                    units, setting, version, grid=grid_name
                )
                det = detector_diagnostics(units, setting, version, grid=grid_name)
                if det:
                    out["detector"][f"{grid_name}|{setting}|{version}"] = det

    path = os.path.join(config.RESULTS_DIR, "summary.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    print(f"已写出 {path}（{len(units)} 个单元，{len(out['grids'])} 个网格）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
