"""决策层方法。

分两类：
  * prompt 级：改变进入上下文的行（os / smote / balctx），需要重新前向。
  * score 级：复用同一份冻结概率，零额外前向开销。

关于 DistPFN：严格按官方仓库 seunghan96/DistPFN 的 README 实现——
调整因子用的是**测试批次平均预测分布** P_test_avg，而非逐行后验。
这意味着 DistPFN 与本文 PICL 属于同一族（类别先验校正），
区别只在目标先验 π* 的取法：
    PICL        π* = 均匀 1/2
    DistPFN     π* = 模型自身在测试批次上的平均预测分布
    DistPFN-T   π* = softmax(P_test_avg / τ)，τ = CE(P_test_avg, π_train)
    PICL-oracle π* = 测试集真实患病率（读标签，非可部署，仅诊断）
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors

EPS = 1e-8


# ------------------------------------------------------------------ 工具


def softmax_temperature(p: np.ndarray, temperature: float) -> np.ndarray:
    """对概率向量做温度缩放的 softmax（DistPFN-T 官方写法）。"""
    z = np.asarray(p, dtype=float) / max(float(temperature), EPS)
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def cross_entropy(p: np.ndarray, q: np.ndarray) -> float:
    """CE(p, q) = -Σ p log q，对应 DistPFN-T 的 τ。"""
    p = np.asarray(p, dtype=float)
    q = np.clip(np.asarray(q, dtype=float), EPS, 1.0)
    return float(-np.sum(p * np.log(q)))


def minority_class(y: np.ndarray) -> int:
    counts = np.bincount(np.asarray(y, dtype=int), minlength=2)
    return int(np.argmin(counts))


# ------------------------------------------------------------------ 先验校正族


def prior_correct(
    proba: np.ndarray, prior_train: np.ndarray, target_prior: np.ndarray
) -> np.ndarray:
    """Elkan(2001) / Saerens et al.(2002) 闭式类别先验校正：

        p*(y=c|x) ∝ q(y=c|x) · π*_c / π_c

    二分类下该映射对 q 严格单调，故 AUROC 不变。
    """
    factor = np.clip(np.asarray(target_prior, dtype=float), EPS, None) / np.clip(
        np.asarray(prior_train, dtype=float), EPS, None
    )
    out = np.asarray(proba, dtype=float) * factor
    row_sum = out.sum(axis=1, keepdims=True)
    return out / np.where(row_sum <= 0, 1.0, row_sum)


def method_picl(proba: np.ndarray, prior_train: np.ndarray, **_) -> np.ndarray:
    """目标先验 = 均匀（论文原 PICL-Prior）。"""
    return prior_correct(proba, prior_train, np.full(proba.shape[1], 1.0 / proba.shape[1]))


def method_distpfn(proba: np.ndarray, prior_train: np.ndarray, **_) -> np.ndarray:
    """DistPFN：π* = 测试批次平均预测分布。"""
    p_test_avg = np.asarray(proba, dtype=float).mean(axis=0)
    return prior_correct(proba, prior_train, p_test_avg)


def method_distpfn_t(proba: np.ndarray, prior_train: np.ndarray, **_) -> np.ndarray:
    """DistPFN-T：π* = softmax(P_test_avg / τ)，τ = CE(P_test_avg, π_train)。"""
    p_test_avg = np.asarray(proba, dtype=float).mean(axis=0)
    tau = cross_entropy(p_test_avg, prior_train)
    return prior_correct(proba, prior_train, softmax_temperature(p_test_avg, tau))


def method_picl_oracle(
    proba: np.ndarray, prior_train: np.ndarray, y_te: np.ndarray | None = None, **_
) -> np.ndarray:
    """π* = 测试集真实患病率。读取测试标签，仅作诊断，非可部署方法。"""
    if y_te is None:
        raise ValueError("picl_oracle 需要 y_te")
    prior_test = np.bincount(np.asarray(y_te, dtype=int), minlength=proba.shape[1]) / len(y_te)
    return prior_correct(proba, prior_train, prior_test)


# ------------------------------------------------------------------ 阈值 / 温度


def _macro_f1_at_threshold(scores: np.ndarray, y: np.ndarray, minority: int, thr: float) -> float:
    """scores 为少数类概率；pred=1 表示判为少数类。"""
    pred = (np.asarray(scores, dtype=float) >= thr).astype(int)
    is_min = (np.asarray(y, dtype=int) == minority).astype(int)
    tp = float(np.sum((pred == 1) & (is_min == 1)))
    fp = float(np.sum((pred == 1) & (is_min == 0)))
    fn = float(np.sum((pred == 0) & (is_min == 1)))
    if tp + fp == 0 or tp + fn == 0:
        return 0.0
    prec, rec = tp / (tp + fp), tp / (tp + fn)
    f_pos = 0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)
    tn = float(np.sum((pred == 0) & (is_min == 0)))
    prec_n = tn / (tn + fn) if tn + fn > 0 else 0.0
    rec_n = tn / (tn + fp) if tn + fp > 0 else 0.0
    f_neg = 0.0 if prec_n + rec_n == 0 else 2 * prec_n * rec_n / (prec_n + rec_n)
    return 0.5 * (f_pos + f_neg)


def tune_threshold(scores_ho: np.ndarray, y_ho: np.ndarray, minority: int) -> float:
    """在训练留出片上搜索最大化 macro-F1 的阈值。"""
    if len(np.unique(y_ho)) < 2:
        return 0.5
    cands = np.unique(np.round(scores_ho, 4))
    if len(cands) > 200:
        cands = np.quantile(scores_ho, np.linspace(0.01, 0.99, 200))
        cands = np.unique(np.round(cands, 4))
    best_t, best_f1 = 0.5, -1.0
    for t in cands:
        f1 = _macro_f1_at_threshold(scores_ho, y_ho, minority, float(t))
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t


def apply_threshold(scores: np.ndarray, minority: int, thr: float) -> np.ndarray:
    pred = (scores >= thr).astype(int)
    out = np.zeros((len(scores), 2), dtype=float)
    out[:, minority] = pred
    out[:, 1 - minority] = 1 - pred
    return out


def tune_temperature(proba_ho: np.ndarray, y_ho: np.ndarray) -> float:
    """在留出片上用对数损失搜索温度。"""
    from scipy.optimize import minimize_scalar

    p = np.clip(proba_ho, EPS, 1.0)

    def nll(log_t: float) -> float:
        t = float(np.exp(log_t))
        z = np.log(p) / t
        z = z - z.max(axis=1, keepdims=True)
        lp = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
        return float(-np.mean(lp[np.arange(len(y_ho)), y_ho.astype(int)]))

    try:
        res = minimize_scalar(nll, bounds=(np.log(0.05), np.log(20.0)), method="bounded")
        return float(np.exp(res.x)) if res.success else 1.0
    except Exception:  # noqa: BLE001
        return 1.0


def apply_temperature(proba: np.ndarray, temperature: float) -> np.ndarray:
    p = np.clip(proba, EPS, 1.0)
    z = np.log(p) / max(temperature, EPS)
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


# ------------------------------------------------------------------ prompt 级


def oversample(X: np.ndarray, y: np.ndarray, seed: int):
    """随机复制少数类行直到各类计数等于多数类计数。"""
    rng = np.random.default_rng(seed)
    counts = np.bincount(y, minlength=2)
    minority = int(np.argmin(counts))
    target = int(counts.max())
    min_idx = np.where(y == minority)[0]
    add = rng.choice(min_idx, size=target - len(min_idx), replace=True)
    keep = np.concatenate([np.arange(len(y)), add])
    return X[keep], y[keep]


def smote_interpolate(X: np.ndarray, y: np.ndarray, seed: int, k: int = 5):
    """因子化矩阵上的局部 k-NN 插值（不使用 imblearn，与论文一致）。"""
    rng = np.random.default_rng(seed)
    counts = np.bincount(y, minlength=2)
    minority = int(np.argmin(counts))
    target = int(counts.max())
    min_idx = np.where(y == minority)[0]
    need = target - len(min_idx)
    if need <= 0 or len(min_idx) < 2:
        return X, y

    Xm = X[min_idx]
    kk = min(k, len(min_idx) - 1)
    nn = NearestNeighbors(n_neighbors=kk + 1).fit(Xm)
    _, nbrs = nn.kneighbors(Xm)
    nbrs = nbrs[:, 1:]

    pick = rng.integers(0, len(min_idx), size=need)
    part = rng.integers(0, kk, size=need)
    lam = rng.random(need)[:, None]
    base = Xm[pick]
    mate = Xm[nbrs[pick, part]]
    synth = base + lam * (mate - base)

    X_new = np.vstack([X, synth])
    y_new = np.concatenate([y, np.full(need, minority, dtype=int)])
    return X_new, y_new


def balance_context(X: np.ndarray, y: np.ndarray, seed: int):
    """把每个类下采样到少数类计数（PICL-BalCtx）。"""
    rng = np.random.default_rng(seed)
    counts = np.bincount(y, minlength=2)
    quota = int(counts.min())
    parts = []
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        parts.append(rng.choice(idx, size=min(quota, len(idx)), replace=False))
    keep = np.sort(np.concatenate(parts))
    return X[keep], y[keep]


# ------------------------------------------------------------------ 非 TabPFN 基线


def build_baselines(seed: int) -> dict:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression as LR

    models = {
        "logreg": LR(class_weight="balanced", max_iter=2000, random_state=seed),
        "rf": RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1),
    }
    try:
        from xgboost import XGBClassifier

        models["xgboost"] = XGBClassifier(
            n_estimators=400,
            max_depth=6,
            learning_rate=0.1,
            random_state=seed,
            eval_metric="logloss",
            n_jobs=-1,
            verbosity=0,
        )
    except Exception:  # noqa: BLE001
        pass
    try:
        from catboost import CatBoostClassifier

        models["catboost"] = CatBoostClassifier(
            iterations=500, depth=6, learning_rate=0.1, random_seed=seed, verbose=False
        )
    except Exception:  # noqa: BLE001
        pass
    try:
        from lightgbm import LGBMClassifier

        models["lightgbm"] = LGBMClassifier(
            n_estimators=400, learning_rate=0.1, random_state=seed, verbose=-1, n_jobs=-1
        )
    except Exception:  # noqa: BLE001
        pass
    return models


def inverse_frequency_weights(y: np.ndarray) -> np.ndarray:
    counts = np.bincount(np.asarray(y, dtype=int), minlength=2).astype(float)
    counts = np.where(counts == 0, 1.0, counts)
    w_per_class = counts.sum() / (len(counts) * counts)
    return w_per_class[np.asarray(y, dtype=int)]


__all__ = [
    "prior_correct",
    "method_picl",
    "method_distpfn",
    "method_distpfn_t",
    "method_picl_oracle",
    "tune_threshold",
    "apply_threshold",
    "tune_temperature",
    "apply_temperature",
    "oversample",
    "smote_interpolate",
    "balance_context",
    "build_baselines",
    "inverse_frequency_weights",
    "minority_class",
    "cross_entropy",
    "softmax_temperature",
]
