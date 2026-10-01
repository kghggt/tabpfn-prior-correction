"""实验执行器。

产物隔离：只写 results/units/ 与 logs/，绝不写基线目录。
断点续跑：每个 (dataset, seed, setting, version) 单元写一个 JSON，已存在则跳过。
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback
from datetime import datetime

import numpy as np
from sklearn.model_selection import train_test_split

import config
import methods as M
from data import build_split
from metrics import evaluate

UNITS_DIR = os.path.join(config.RESULTS_DIR, "units")
os.makedirs(UNITS_DIR, exist_ok=True)

_LOG_PATH = os.path.join(
    config.LOGS_DIR, f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
)


def log(msg: str) -> None:
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(_LOG_PATH, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


# ------------------------------------------------------------------ TabPFN


_MODEL_CACHE: dict = {}


def get_tabpfn(version: str, seed: int):
    """按版本构造 TabPFN 分类器；模型权重按版本缓存复用。"""
    key = (version, seed)
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]

    from tabpfn import TabPFNClassifier

    kwargs = dict(
        n_estimators=config.N_ESTIMATORS,
        ignore_pretraining_limits=config.IGNORE_PRETRAINING_LIMITS,
        random_state=seed,
        device="cuda",
    )
    # 显式本地权重路径：TabPFNClassifier(model_path="v2") 会把 "v2" 当作文件名，
    # 必须传真实路径才能正确选版。
    clf = TabPFNClassifier(model_path=config.model_path(version), **kwargs)
    _MODEL_CACHE[key] = clf
    return clf


# 记录最近一次前向是否走了降级路径；由 run_split 写入产物以便追溯
_LAST_MODE: dict = {"degraded": None}


def release_cuda(clf=None) -> None:
    """释放上一次前向占用的显存。

    TabPFN 每次 fit 都会新建 inference engine，旧 engine 若不被显式丢弃，
    显存会随单元累积；而 CUDA 的异步报错会把真实的 OOM 推迟到下一个
    fit 才抛出来，表现为"小表也 OOM"。所以每个单元结束必须清一次。
    """
    try:
        import gc

        import torch

        # 注意：不能把 executor_ / ensemble_preprocessor_ 置 None——tabpfn 在
        # 下次 fit 时会先 estimator_to_device(estimator.executor_)，None 会直接
        # AttributeError。交给 gc 回收即可：fit() 每次都重建这两个属性。
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            try:
                torch.cuda.synchronize()
            except Exception:  # noqa: BLE001
                pass
    except Exception:  # noqa: BLE001
        pass


def fit_predict_proba(clf, X_tr, y_tr, X_te, version: str | None = None) -> np.ndarray:
    """带 OOM 回退的前向。

    回退顺序：
      1. predict 阶段的 OOM 用**测试集分块**解决。分块不改变任何模型配置，
         结果与整块前向逐行一致，因此不会污染对比。
      2. fit 阶段就 OOM 时逐级降级（少 estimator → 单 estimator → CPU），
         且必须显式带本地权重路径，否则 tabpfn 会去拉默认的 v3.5 gated 仓库
         （无 token 直接报错而不是回退）。
    """
    try:
        clf.fit(X_tr, y_tr)
    except Exception as exc:  # noqa: BLE001
        msg = str(exc).lower()
        if "memory" not in msg and "oom" not in msg:
            raise
        log(f"    fit 阶段 OOM，开始降级: {type(exc).__name__}")
        from tabpfn import TabPFNClassifier

        # 逐级降级：少 estimator -> 单 estimator -> CPU。
        # 每级都必须显式带本地权重路径（不带会去拉 gated 默认仓库，无 token 直接报错）。
        # 注意：fit_mode="batched" 会在 fit() 里被强制改回 "fit_preprocessors"，
        # 所以真正能省显存的是 "low_memory"（按批重算预处理，不缓存 KV）。
        stages = [
            dict(n_estimators=2, device="cuda", memory_saving_mode=2, fit_mode="low_memory"),
            dict(n_estimators=1, device="cuda", memory_saving_mode=2, fit_mode="low_memory"),
        ]
        if config.ALLOW_CPU_FALLBACK:
            stages.append(dict(n_estimators=1, device="cpu", memory_saving_mode=2, fit_mode="batched"))
        last_err = None
        for st in stages:
            try:
                import torch

                torch.cuda.empty_cache()
                fallback = TabPFNClassifier(
                    model_path=config.model_path(version) if version else None,
                    ignore_pretraining_limits=config.IGNORE_PRETRAINING_LIMITS,
                    random_state=0,
                    **st,
                )
                fallback.fit(X_tr, y_tr)
                log(f"    降级成功: {st}")
                _LAST_MODE["degraded"] = f"fit:{st['device']}/n{st['n_estimators']}"
                try:
                    return np.asarray(fallback.predict_proba(X_te), dtype=float)
                except Exception as exc2:  # noqa: BLE001
                    if "memory" not in str(exc2).lower() and "oom" not in str(exc2).lower():
                        raise
                    step = int(os.environ.get("TABPFN_CHUNK", "256"))
                    chunks = [
                        np.asarray(fallback.predict_proba(X_te[i : i + step]), dtype=float)
                        for i in range(0, len(X_te), step)
                    ]
                    return np.vstack(chunks)
            except Exception as exc2:  # noqa: BLE001
                last_err = exc2
                log(f"    降级失败 {st}: {str(exc2)[:80]}")
        raise RuntimeError(f"fit 阶段所有降级方案均失败: {last_err!r}")

    try:
        return np.asarray(clf.predict_proba(X_te), dtype=float)
    except Exception as exc:  # noqa: BLE001
        msg = str(exc).lower()
        if "memory" not in msg and "oom" not in msg:
            raise
        step = int(os.environ.get("TABPFN_CHUNK", "256"))
        log(f"    predict 阶段 OOM（{len(X_te)} 行），分块前向 step={step}: {type(exc).__name__}")
        chunks = []
        for i in range(0, len(X_te), step):
            chunks.append(np.asarray(clf.predict_proba(X_te[i : i + step]), dtype=float))
        return np.vstack(chunks)


# ------------------------------------------------------------------ 单元执行


def cap_prompt(X: np.ndarray, y: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """把 prompt 级方法放大后的上下文压到 HARD_CONTEXT_CAP 以内（分层）。"""
    if len(y) <= config.HARD_CONTEXT_CAP:
        return X, y
    rng = np.random.default_rng(seed + 20_000)
    keep = []
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        share = max(1, int(round(config.HARD_CONTEXT_CAP * len(idx) / len(y))))
        keep.append(rng.choice(idx, size=min(share, len(idx)), replace=False))
    keep = np.sort(np.concatenate(keep))
    return X[keep], y[keep]


def unit_filename(dataset: str, seed: int, setting: str, version: str, grid: str) -> str:
    """单元文件名。

    为什么需要这个函数：wilt / pc1 / kc1 同时属于 main12 与 rare12，
    两边的 (dataset, seed, natural) 完全同名，若文件名只由这三者决定，
    后跑的网格会把前一个网格的单元当作"已完成"直接跳过，导致其中一个网格
    静默缺表。所以除主网格外，文件名一律带网格前缀。
    """
    suffix = f"__{config.RUN_TAG}" if config.RUN_TAG else ""
    if grid == "main12" and not suffix:
        return f"{dataset}__{seed}__{setting}__{version}.json"
    return f"{dataset}__{seed}__{setting}__{version}__{grid}{suffix}.json"


def run_unit(dataset: str, seed: int, setting: str, version: str, grid: str | None = None) -> dict | None:
    """grid 为空时取 config.RUN_TAG（消融变体），否则默认 main12。"""
    grid = grid or (config.RUN_TAG or "main12")
    out_path = os.path.join(UNITS_DIR, unit_filename(dataset, seed, setting, version, grid))
    if os.path.exists(out_path):
        return None  # 已完成，跳过
    split = build_split(dataset, seed, setting)
    return run_split(split, version, seed, out_path, grid=grid)


def run_split(split: dict, version: str, seed: int, out_path: str, grid: str = "main12") -> dict:
    """对任意已构造好的切分跑全部方法。

    grid 字段用于聚合时分组；自然偏移场景传 "shift"，扩展网格传 "extended"。
    """
    if os.path.exists(out_path):
        return None  # 已完成，跳过

    t0 = time.time()
    X_tr, y_tr = split["X_tr"], split["y_tr"]
    X_te, y_te = split["X_te"], split["y_te"]
    minority = split["minority_class"]
    prior_train = split["prior_train"]

    clf = get_tabpfn(version, seed)
    results: dict[str, dict] = {}

    # ---- prompt 级方法（各自需要一次前向）
    _LAST_MODE["degraded"] = None
    P_vanilla = fit_predict_proba(clf, X_tr, y_tr, X_te, version)
    forward_mode = _LAST_MODE["degraded"]
    results["tabpfn"] = evaluate(P_vanilla, y_te, minority)

    X_os, y_os = M.oversample(X_tr, y_tr, seed)
    X_os, y_os = cap_prompt(X_os, y_os, seed)
    P_os = fit_predict_proba(clf, X_os, y_os, X_te, version)
    results["os"] = evaluate(P_os, y_te, minority)

    try:
        X_sm, y_sm = M.smote_interpolate(X_tr, y_tr, seed, k=config.SMOTE_K)
        X_sm, y_sm = cap_prompt(X_sm, y_sm, seed)
        P_sm = fit_predict_proba(clf, X_sm, y_sm, X_te, version)
        results["smote"] = evaluate(P_sm, y_te, minority)
    except Exception as exc:  # noqa: BLE001
        log(f"    SMOTE 失败，跳过: {exc!r}")

    X_bc, y_bc = M.balance_context(X_tr, y_tr, seed)
    if len(np.unique(y_bc)) == 2:
        P_bc = fit_predict_proba(clf, X_bc, y_bc, X_te, version)
        prior_bc = np.bincount(y_bc, minlength=2) / len(y_bc)
        results["balctx"] = evaluate(
            M.prior_correct(P_bc, prior_bc, np.array([0.5, 0.5])), y_te, minority
        )
        results["balctx_raw"] = evaluate(P_bc, y_te, minority)

    # ---- 阈值 / 温度的留出片（额外一次前向）
    thr = 0.5
    temperature = 1.0
    holdout_diag = None  # 可部署的先验保真度诊断（只用训练留出片，不碰测试标签）
    clf_ho = None
    try:
        if (np.bincount(y_tr, minlength=2) >= 2).all() and len(y_tr) >= 20:
            ho_idx, fit_idx = train_test_split(
                np.arange(len(y_tr)),
                test_size=config.HOLDOUT_FRACTION,
                random_state=seed,
                stratify=y_tr,
            )
            clf_ho = get_tabpfn(version, seed)
            clf_ho.fit(X_tr[fit_idx], y_tr[fit_idx])
            P_ho = np.asarray(clf_ho.predict_proba(X_tr[ho_idx]), dtype=float)
            thr = M.tune_threshold(P_ho[:, minority], y_tr[ho_idx], minority)
            temperature = M.tune_temperature(P_ho, y_tr[ho_idx])
            ho_pred = (P_ho[:, minority] >= 0.5).astype(int)
            true_rate = float(np.mean(y_tr[ho_idx] == minority))
            pred_rate = float(np.mean(ho_pred))
            holdout_diag = {
                "true_minority_rate": true_rate,
                "predicted_minority_rate": pred_rate,
                "fidelity": (pred_rate / true_rate) if true_rate > 0 else float("nan"),
                "n_holdout": int(len(ho_idx)),
            }
    except Exception as exc:  # noqa: BLE001
        log(f"    留出片调参失败，使用默认值: {exc!r}")

    results["thr"] = evaluate(
        M.apply_threshold(P_vanilla[:, minority], minority, thr), y_te, minority
    )

    # ---- score 级方法（复用同一份冻结概率）
    results["picl"] = evaluate(M.method_picl(P_vanilla, prior_train), y_te, minority)
    results["distpfn"] = evaluate(M.method_distpfn(P_vanilla, prior_train), y_te, minority)
    results["distpfn_t"] = evaluate(M.method_distpfn_t(P_vanilla, prior_train), y_te, minority)
    results["picl_oracle"] = evaluate(
        M.method_picl_oracle(P_vanilla, prior_train, y_te=y_te), y_te, minority
    )
    P_temp = M.apply_temperature(P_vanilla, temperature)
    results["picl_full"] = evaluate(
        M.method_picl(P_temp, prior_train), y_te, minority
    )

    # ---- 非 TabPFN 基线
    for name, model in M.build_baselines(seed).items():
        try:
            model.fit(X_tr, y_tr)
            P = np.asarray(model.predict_proba(X_te), dtype=float)
            results[name] = evaluate(P, y_te, minority)
        except Exception as exc:  # noqa: BLE001
            log(f"    基线 {name} 失败: {exc!r}")
            continue

        if name in ("xgboost", "catboost", "lightgbm"):
            try:
                import copy

                wmodel = copy.deepcopy(model)
                w = M.inverse_frequency_weights(y_tr)
                wmodel.fit(X_tr, y_tr, sample_weight=w)
                P = np.asarray(wmodel.predict_proba(X_te), dtype=float)
                results[f"{name}_w"] = evaluate(P, y_te, minority)
            except Exception as exc:  # noqa: BLE001
                log(f"    加权基线 {name} 失败: {exc!r}")

    payload = {
        "dataset": split["dataset"],
        "grid": grid,
        "openml_id": split.get("openml_id"),
        "seed": int(seed),
        "setting": split.get("setting", "natural"),
        "shift_descriptor": split.get("shift_descriptor"),
        "model_version": version,
        "minority_class": minority,
        "n_train": split["n_train"],
        "n_test": split["n_test"],
        "n_features": split.get("n_features"),
        "context_cap": split.get("context_cap"),
        "prior_train": [float(x) for x in split["prior_train"]],
        "prior_train_full": [float(x) for x in split["prior_train_full"]],
        "prior_test": [float(x) for x in split["prior_test"]],
        "chosen_threshold": float(thr),
        "chosen_temperature": float(temperature),
        "forward_mode": forward_mode,  # None 表示默认配置；否则记录降级方式
        "holdout_diag": holdout_diag,
        "methods": results,
        "runtime_sec": round(time.time() - t0, 1),
    }
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=1)
    release_cuda(clf)
    release_cuda(clf_ho)
    return payload


def run_grid(datasets, seeds, settings, versions, tag: str = "", grid: str = "main12") -> None:
    total = len(datasets) * len(seeds) * len(settings) * len(versions)
    done = n = 0
    log(f"开始：{len(datasets)} 表 × {len(seeds)} 种子 × {len(settings)} 设定 × {len(versions)} 版本 = {total} 单元 {tag}")
    for ds in datasets:
        for sd in seeds:
            for st in settings:
                for ver in versions:
                    n += 1
                    try:
                        payload = run_unit(ds, sd, st, ver, grid=grid)
                        if payload is None:
                            done += 1
                            log(f"[{n}/{total}] {ds} {st} seed{sd} {ver} — 已存在，跳过")
                            continue
                        mf1 = payload["methods"]["tabpfn"]["macro_f1"]
                        picl = payload["methods"]["picl"]["macro_f1"]
                        dist = payload["methods"]["distpfn"]["macro_f1"]
                        log(
                            f"[{n}/{total}] {ds} {st} seed{sd} {ver} — "
                            f"tabpfn {mf1:.3f} / picl {picl:.3f} / distpfn {dist:.3f} "
                            f"({payload['runtime_sec']}s)"
                        )
                    except Exception as exc:  # noqa: BLE001
                        msg = str(exc).lower()
                        log(f"[{n}/{total}] {ds} {st} seed{sd} {ver} — 失败: {exc!r}")
                        # 显存放不下时，把上下文预算折半重跑该单元（同一单元内所有方法
                        # 共用缩小后的上下文，方法间比较仍然配对、仍然公平）。
                        if ("memory" in msg or "oom" in msg) and config.BUDGET_SCALE > 0.2:
                            old = config.BUDGET_SCALE
                            config.BUDGET_SCALE = max(0.2, old / 2.0)
                            log(f"    cells 预算 x{old} -> x{config.BUDGET_SCALE}，重试 {ds} {st} seed{sd} {ver}")
                            release_cuda()
                            try:
                                payload = run_unit(ds, sd, st, ver, grid=grid)
                                if payload is not None:
                                    log(f"    重试成功: tabpfn {payload['methods']['tabpfn']['macro_f1']:.3f}")
                            except Exception as exc2:  # noqa: BLE001
                                log(f"    重试仍失败: {exc2!r}")
                            finally:
                                config.BUDGET_SCALE = old
                                release_cuda()
                        else:
                            log(traceback.format_exc())
                    finally:
                        release_cuda()
    log(f"结束：{total} 单元，{done} 个命中缓存")
