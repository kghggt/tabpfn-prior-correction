"""指标。主指标 macro-F1，辅以 balanced accuracy、AUROC、预测少数类率。"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    classes = np.unique(np.concatenate([y_true, y_pred]))
    scores = []
    for c in classes:
        tp = float(np.sum((y_pred == c) & (y_true == c)))
        fp = float(np.sum((y_pred == c) & (y_true != c)))
        fn = float(np.sum((y_pred != c) & (y_true == c)))
        if tp + fp == 0 or tp + fn == 0:
            scores.append(0.0)
            continue
        prec, rec = tp / (tp + fp), tp / (tp + fn)
        scores.append(0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec))
    return float(np.mean(scores))


def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    classes = np.unique(y_true)
    per = []
    for c in classes:
        mask = y_true == c
        if mask.sum() == 0:
            continue
        per.append(float(np.mean(y_pred[mask] == c)))
    return float(np.mean(per))


def auroc(y_true: np.ndarray, scores_minority: np.ndarray, minority: int) -> float:
    y_true = np.asarray(y_true, dtype=int)
    y_bin = (y_true == minority).astype(int)
    if len(np.unique(y_bin)) < 2:
        return float("nan")
    return float(roc_auc_score(y_bin, np.asarray(scores_minority, dtype=float)))


def predicted_minority_rate(proba: np.ndarray, minority: int) -> float:
    pred = np.argmax(np.asarray(proba, dtype=float), axis=1)
    return float(np.mean(pred == minority))


def expected_calibration_error(proba: np.ndarray, y_true: np.ndarray, n_bins: int = 15) -> float:
    conf = np.asarray(proba, dtype=float).max(axis=1)
    pred = np.argmax(np.asarray(proba, dtype=float), axis=1)
    correct = (pred == np.asarray(y_true, dtype=int)).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum() == 0:
            continue
        ece += abs(correct[m].mean() - conf[m].mean()) * (m.sum() / len(conf))
    return float(ece)


def evaluate(proba: np.ndarray, y_true: np.ndarray, minority: int) -> dict:
    """proba 为 (n, C) 概率矩阵；AUROC 用原始少数类概率。"""
    proba = np.asarray(proba, dtype=float)
    pred = proba.argmax(axis=1)
    return {
        "macro_f1": macro_f1(y_true, pred),
        "balanced_accuracy": balanced_accuracy(y_true, pred),
        "auroc": auroc(y_true, proba[:, minority], minority),
        "predicted_minority_rate": predicted_minority_rate(proba, minority),
        "accuracy": float(np.mean(pred == np.asarray(y_true, dtype=int))),
        "ece": expected_calibration_error(proba, y_true),
    }


__all__ = [
    "macro_f1",
    "balanced_accuracy",
    "auroc",
    "predicted_minority_rate",
    "expected_calibration_error",
    "evaluate",
]
