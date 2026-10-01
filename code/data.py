"""数据装载与切分。

OpenML 数据集按名称解析；分类列在切分前一次性因子化（与论文一致）。
诱导设定只下采样**训练侧**少数类，测试集从不被重采样。
"""

from __future__ import annotations

import json
import os
import re
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import config

from config import (
    INDUCED_RATIO,
    MAX_CONTEXT_ROWS,
    MAX_TEST_ROWS,
    MIN_MINORITY_ROWS,
    RESULTS_DIR,
)

warnings.filterwarnings("ignore")

INDEX_PATH = os.path.join(RESULTS_DIR, "openml_index.json")


def _build_index(refresh: bool = False) -> dict:
    """把 OpenML 数据集名称解析为 id，结果缓存到 results/。"""
    if os.path.exists(INDEX_PATH) and not refresh:
        with open(INDEX_PATH, "r", encoding="utf-8") as fh:
            return json.load(fh)

    import openml

    listing = openml.datasets.list_datasets(output_format="dataframe")
    index: dict[str, int] = {}
    for did, row in listing.iterrows():
        name = str(row.get("name", "")).strip()
        if not name:
            continue
        key = name.lower()
        # 同名时保留 id 最小的那个，保证解析稳定可复现
        if key not in index or int(did) < index[key]:
            index[key] = int(did)

    with open(INDEX_PATH, "w", encoding="utf-8") as fh:
        json.dump(index, fh, indent=0, sort_keys=True)
    return index


def resolve_dataset_id(name: str, index: dict | None = None) -> int | None:
    """按名称（忽略大小写与分隔符差异）解析 OpenML dataset id。"""
    index = index or _build_index()
    key = name.strip().lower()
    if key in config.DATASET_ID_OVERRIDE:
        return config.DATASET_ID_OVERRIDE[key]
    if key in index:
        return index[key]
    norm = re.sub(r"[^a-z0-9]+", "", key)
    for k, v in index.items():
        if re.sub(r"[^a-z0-9]+", "", k) == norm:
            return v
    return None


def load_openml_binary(name: str):
    """返回 (X, y, meta)；X 为 float 矩阵（分类列已因子化），y 为 0/1。"""
    import openml

    did = resolve_dataset_id(name)
    if did is None:
        raise KeyError(f"OpenML 上找不到数据集: {name}")
    ds = openml.datasets.get_dataset(
        did, download_data=True, download_qualities=False, download_features_meta_data=False
    )
    target = ds.default_target_attribute
    X_df, y, _, _ = ds.get_data(target=target)

    if isinstance(y, pd.Series):
        y_raw = y.values
    else:
        y_raw = np.asarray(y)

    # 只保留二分类
    classes = pd.unique(pd.Series(y_raw).dropna())
    if len(classes) != 2:
        raise ValueError(f"{name}: 非二分类（{len(classes)} 类）")

    y = pd.factorize(pd.Series(y_raw))[0].astype(int)

    X_df = pd.DataFrame(X_df).reset_index(drop=True)
    for col in X_df.columns:
        if not pd.api.types.is_numeric_dtype(X_df[col]):
            X_df[col] = pd.factorize(X_df[col].astype(str))[0]
    X_df = X_df.replace([np.inf, -np.inf], np.nan)
    X = X_df.fillna(0.0).to_numpy(dtype=float)

    meta = {
        "dataset": name,
        "openml_id": int(did),
        "n_rows": int(X.shape[0]),
        "n_features": int(X.shape[1]),
    }
    return X, y, meta


def load_openml_raw(name: str):
    """返回未因子化的 (X_df, y, meta)，供子群漂移切分使用。"""
    import openml

    did = resolve_dataset_id(name)
    if did is None:
        raise KeyError(f"OpenML 上找不到数据集: {name}")
    ds = openml.datasets.get_dataset(
        did, download_data=True, download_qualities=False, download_features_meta_data=False
    )
    X_df, y, _, _ = ds.get_data(target=ds.default_target_attribute)
    X_df = pd.DataFrame(X_df).reset_index(drop=True)
    if isinstance(y, pd.Series):
        y_raw = y.values
    else:
        y_raw = np.asarray(y)
    classes = pd.unique(pd.Series(y_raw).dropna())
    if len(classes) != 2:
        raise ValueError(f"{name}: 非二分类（{len(classes)} 类）")
    y = pd.factorize(pd.Series(y_raw))[0].astype(int)
    meta = {"dataset": name, "openml_id": int(did)}
    return X_df, y, meta


def _factorize_frame(X_df: pd.DataFrame) -> np.ndarray:
    out = pd.DataFrame(X_df).copy()
    for col in out.columns:
        if not pd.api.types.is_numeric_dtype(out[col]):
            out[col] = pd.factorize(out[col].astype(str))[0]
    out = out.replace([np.inf, -np.inf], np.nan)
    return out.fillna(0.0).to_numpy(dtype=float)


def pick_subgroup_pair(X_df: pd.DataFrame, y: np.ndarray, min_group: int = 250):
    """挑选天然患病率差异最大的子群对。

    只看基数 2..6 的列，每组至少 min_group 行。
    返回 (col, train_values, test_values, prev_train, prev_test)；找不到返回 None。
    方向固定为"训练侧稀有类更少、测试侧更多"，即必须放大被低估的稀有类。
    """
    best = None
    for col in X_df.columns:
        s = X_df[col]
        vals = pd.Series(s).astype(str)
        nuniq = vals.nunique()
        if not (2 <= nuniq <= 6):
            continue
        counts = vals.value_counts()
        groups = [v for v, c in counts.items() if c >= min_group]
        if len(groups) < 2:
            continue
        prev = {}
        for g in groups:
            mask = (vals == g).to_numpy()
            yy = y[mask]
            if len(np.unique(yy)) < 2:
                continue
            prev[g] = float(np.bincount(yy).min() / len(yy))
        if len(prev) < 2:
            continue
        items = sorted(prev.items(), key=lambda kv: kv[1])
        lo, hi = items[0], items[-1]
        if lo[0] == hi[0]:
            continue
        gap = hi[1] - lo[1]
        if best is None or gap > best[0]:
            best = (gap, col, [lo[0]], [hi[0]], lo[1], hi[1])
    if best is None:
        return None
    _, col, tr_vals, te_vals, p_tr, p_te = best
    return col, tr_vals, te_vals, p_tr, p_te


def build_subgroup_split(
    name: str,
    seed: int,
    cap_test: int = 1000,
    cap_train: int = 2048,
):
    """构造天然先验偏移切分：训练用低患病率子群，测试用高患病率子群。

    不做任何标签重采样；先验差异来自数据本身的亚群结构。
    """
    X_df, y, meta = load_openml_raw(name)
    picked = pick_subgroup_pair(X_df, y)
    if picked is None:
        raise ValueError(f"{name}: 找不到满足条件的子群对")
    col, tr_vals, te_vals, _, _ = picked

    vals = X_df[col].astype(str).to_numpy()
    tr_mask = np.isin(vals, tr_vals)
    te_mask = np.isin(vals, te_vals)
    if tr_mask.sum() < 50 or te_mask.sum() < 50:
        raise ValueError(f"{name}: 子群样本不足")

    # 从特征矩阵中剔除分组列本身，避免把分组变量当成输入特征泄漏
    feat_df = X_df.drop(columns=[col])
    X_all = _factorize_frame(feat_df)

    X_tr_full, y_tr_full = X_all[tr_mask], y[tr_mask]
    X_te_full, y_te_full = X_all[te_mask], y[te_mask]

    # 显存预算：与 build_split 同一套 cells 预算，测试侧与上下文侧同时让位
    n_features = int(X_all.shape[1])
    cap_test, cap_train = config.fit_caps(len(y_te_full), n_features)
    config.HARD_CONTEXT_CAP = cap_train

    rng = np.random.default_rng(seed)
    if len(y_te_full) > cap_test:
        keep = rng.choice(len(y_te_full), size=cap_test, replace=False)
        keep.sort()
        X_te_full, y_te_full = X_te_full[keep], y_te_full[keep]

    # 6GB 显存下 TabPFN 的上下文是 O(n^2)，子群场景训练集往往上万行，
    # 统一把训练侧压到 cap_train，测试侧压到 cap_test，避免整批掉进 CPU 回退。
    X_tr, y_tr = subsample_context(X_tr_full, y_tr_full, seed, cap=cap_train)
    if len(np.unique(y_tr)) < 2 or len(np.unique(y_te_full)) < 2:
        raise ValueError(f"{name}: 切分后某一侧只剩一个类别")

    counts_tr = np.bincount(y_tr, minlength=2)
    prior_train = counts_tr / counts_tr.sum()
    minority = int(np.argmin(counts_tr))

    return {
        **meta,
        "dataset": name,
        "seed": int(seed),
        "setting": f"sub_{re.sub(r'[^A-Za-z0-9]', '', str(col))[:24]}",
        "X_tr": X_tr,
        "y_tr": y_tr,
        "X_te": X_te_full,
        "y_te": y_te_full,
        "minority_class": minority,
        "prior_train": prior_train,
        "prior_train_full": prior_train,
        "prior_test": np.bincount(y_te_full, minlength=2) / len(y_te_full),
        "n_train": int(len(y_tr)),
        "n_test": int(len(y_te_full)),
        "n_features": int(X_all.shape[1]),
        "context_cap": int(cap_train),
        "shift_descriptor": {
            "kind": "subgroup",
            "column": str(col),
            "train_values": [str(v) for v in tr_vals],
            "test_values": [str(v) for v in te_vals],
        },
    }


def split_indices(y: np.ndarray, seed: int):
    idx = np.arange(len(y))
    tr, te = train_test_split(idx, test_size=0.30, random_state=seed, stratify=y)
    return np.sort(tr), np.sort(te)


def downsample_minority(
    X: np.ndarray, y: np.ndarray, ratio: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """把训练侧少数类下采样到 n_min / n_maj ≈ ratio，测试侧不动。"""
    rng = np.random.default_rng(seed)
    counts = np.bincount(y)
    minority = int(np.argmin(counts))
    majority = 1 - minority
    n_maj = int(counts[majority])

    target_min = int(round(n_maj * ratio / (1.0 - ratio)))
    target_min = max(MIN_MINORITY_ROWS, min(target_min, int(counts[minority])))

    min_idx = np.where(y == minority)[0]
    maj_idx = np.where(y == majority)[0]
    keep_min = rng.choice(min_idx, size=target_min, replace=False)

    keep = np.concatenate([keep_min, maj_idx])
    keep.sort()
    return X[keep], y[keep]


def subsample_context(X: np.ndarray, y: np.ndarray, seed: int, cap: int | None = None):
    """训练侧超过上下文预算时随机子采样（分层）。

    cap=None 时取 config.MAX_CONTEXT_ROWS（在调用时读取，便于运行期下调上下文预算）。
    """
    if cap is None:
        cap = config.MAX_CONTEXT_ROWS
    n = len(y)
    if n <= cap:
        return X, y
    rng = np.random.default_rng(seed + 10_000)
    keep = []
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        share = max(1, int(round(cap * len(idx) / n)))
        keep.append(rng.choice(idx, size=min(share, len(idx)), replace=False))
    keep = np.sort(np.concatenate(keep))
    return X[keep], y[keep]


def subsample_test(X: np.ndarray, y: np.ndarray, seed: int, cap: int | None = None):
    """测试侧超过上限时分层子采样（cap=None 表示不设上限）。

    只对超过上限的表生效，且子采样在切分之后、与训练侧无关，
    因此不改变训练先验，只影响评估样本数。
    """
    if cap is None or cap <= 0:
        return X, y
    n = len(y)
    if n <= cap:
        return X, y
    rng = np.random.default_rng(seed + 30_000)
    keep = []
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        share = max(1, int(round(cap * len(idx) / n)))
        keep.append(rng.choice(idx, size=min(share, len(idx)), replace=False))
    keep = np.sort(np.concatenate(keep))
    return X[keep], y[keep]


def build_split(name: str, seed: int, setting: str):
    """构造一个 (dataset, seed, setting) 的训练/测试划分。

    返回 dict，含 X_tr/y_tr/X_te/y_te 以及先验信息。
    """
    X, y, meta = load_openml_binary(name)
    tr_idx, te_idx = split_indices(y, seed)
    X_tr_full, y_tr_full = X[tr_idx], y[tr_idx]
    X_te, y_te = X[te_idx], y[te_idx]

    # 显存预算：测试侧与上下文侧同时让位（见 config.fit_caps）
    n_features = int(meta.get("n_features", X.shape[1]))
    n_te_cap, ctx_cap = config.fit_caps(len(y_te), n_features)
    X_te, y_te = subsample_test(X_te, y_te, seed, cap=n_te_cap)
    config.HARD_CONTEXT_CAP = ctx_cap  # +OS/+SMOTE 放大后的 prompt 用同一个上限

    if setting in INDUCED_RATIO:
        X_tr, y_tr = downsample_minority(X_tr_full, y_tr_full, INDUCED_RATIO[setting], seed)
    else:
        X_tr, y_tr = X_tr_full, y_tr_full

    X_tr, y_tr = subsample_context(X_tr, y_tr, seed, cap=ctx_cap)

    counts_tr = np.bincount(y_tr, minlength=2)
    prior_train = counts_tr / counts_tr.sum()
    minority = int(np.argmin(counts_tr))

    # 未下采样的原始训练先验，用于报告"诱导偏移强度"
    counts_full = np.bincount(y_tr_full, minlength=2)
    prior_train_full = counts_full / counts_full.sum()

    return {
        **meta,
        "seed": int(seed),
        "setting": setting,
        "X_tr": X_tr,
        "y_tr": y_tr,
        "X_te": X_te,
        "y_te": y_te,
        "minority_class": minority,
        "prior_train": prior_train,
        "prior_train_full": prior_train_full,
        "prior_test": np.bincount(y_te, minlength=2) / len(y_te),
        "n_train": int(len(y_tr)),
        "n_test": int(len(y_te)),
        "context_cap": int(ctx_cap),
    }
