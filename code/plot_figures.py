"""Regenerate the letter's figures from the frozen run artefacts.

Reads only `results/units/*.json` and `results/summary.json` (no hand-typed
numbers) and writes the figures that the submission compiles:

  figures/fig1_rank.pdf    Figure 1  AUROC vs macro-F1 scatter
  figures/fig2_bars.pdf    Figure 2  mean macro-F1 bars (3 settings)
  figures/fig3_prior.pdf   Figure 3  predicted vs test minority rate
  figures/graphical_abstract.{pdf,png}

Figure 4 (fig4_rule.pdf) is produced separately by code/plot_rule.py.
"""

from __future__ import annotations

import glob
import json
import os
import statistics as st

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # repository root
UNITS = os.path.join(ROOT, "results", "units")
SUMMARY = os.path.join(ROOT, "results", "summary.json")
FIGS = os.path.join(ROOT, "figures")
GA = FIGS

MAIN = ["australian", "blood-transfusion", "credit-g", "diabetes", "ilpd", "kc1",
        "pc1", "phoneme", "qsar-biodeg", "spambase", "wdbc", "wilt"]
SEEDS = (0, 1, 2)

GREY, BLUE = "#8c8c8c", "#17557e"
C5, C10, CNAT = "#b2182b", "#e0a03a", "#17557e"
METHOD_COLOR = {"TabPFN": "#8c8c8c", "+OS": "#b2182b", "BalCtx": "#e0a03a",
                "Thr": "#a6a6a6", "PICL": "#17557e", "LogReg": "#2e8b7a", "XGB-w": "#6a3d9a"}


def load(fn):
    with open(fn, encoding="utf-8") as fh:
        return json.load(fh)


def unit(ds, seed, setting, version):
    p = os.path.join(UNITS, f"{ds}__{seed}__{setting}__{version}.json")
    return load(p) if os.path.exists(p) else None


def main_scatter():
    """Figure 1: one point per dataset, AUROC on x, macro-F1 on y, PICL lifts vertically."""
    xs, tab, picl = [], [], []
    for ds in MAIN:
        au = [unit(ds, s, "ind05", "v2")["methods"]["tabpfn"]["auroc"] for s in SEEDS]
        ft = [unit(ds, s, "ind05", "v2")["methods"]["tabpfn"]["macro_f1"] for s in SEEDS]
        fp = [unit(ds, s, "ind05", "v2")["methods"]["picl"]["macro_f1"] for s in SEEDS]
        xs.append(st.fmean(au)); tab.append(st.fmean(ft)); picl.append(st.fmean(fp))
    fig, ax = plt.subplots(figsize=(3.35, 2.75))
    for x, a, b in zip(xs, tab, picl):
        ax.plot([x, x], [a, b], color="0.80", linewidth=0.9, zorder=1)
    ax.scatter(xs, tab, s=24, color=GREY, zorder=3, label="TabPFN")
    ax.scatter(xs, picl, s=24, color=BLUE, zorder=3, label="PICL-Prior")
    ax.plot([0.68, 1.0], [0.68, 1.0], color="0.60", linewidth=0.7, linestyle="--", zorder=0)
    ax.set_xlabel("AUROC (ranking)", fontsize=8)
    ax.set_ylabel("Macro-F1 (decision)", fontsize=8)
    ax.set_xlim(0.70, 1.01); ax.set_ylim(0.36, 1.0)
    ax.set_xticks([0.7, 0.8, 0.9, 1.0])
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=6.5, frameon=False, loc="lower right")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "fig1_rank.pdf"), bbox_inches="tight")
    plt.close(fig)


def main_bars():
    """Figure 2: mean macro-F1 per method under the three settings."""
    order = ["TabPFN", "+OS", "BalCtx", "Thr", "PICL", "LogReg", "XGB-w"]
    key = {"TabPFN": "tabpfn", "+OS": "os", "BalCtx": "balctx", "Thr": "thr",
           "PICL": "picl", "LogReg": "logreg", "XGB-w": "xgboost_w"}
    grids = load(SUMMARY)["grids"]
    settings = [("Induced 5\\%", "main12|ind05|v2", C5),
                ("10\\%", "main12|ind10|v2", C10),
                ("Natural", "main12|natural|v2", CNAT)]
    fig, ax = plt.subplots(figsize=(3.45, 2.75))
    n = len(order); w = 0.26
    for gi, (lbl, gkey, col) in enumerate(settings):
        vals = [grids[gkey]["methods"][key[m]]["mean"] for m in order]
        xs = [i + (gi - 1) * w for i in range(n)]
        ax.bar(xs, vals, width=w, color=col, label=lbl.replace("\\%", "%"), zorder=3)
    ax.set_xticks(range(n)); ax.set_xticklabels(order, fontsize=7, rotation=30, ha="right")
    ax.set_ylabel("Mean macro-F1", fontsize=8)
    ax.set_ylim(0.40, 0.83)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=6.5, frameon=False, ncol=3, loc="upper left")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "fig2_bars.pdf"), bbox_inches="tight")
    plt.close(fig)


def main_prior():
    """Figure 3: predicted minority rate vs the test-fold minority rate."""
    tx, pt, pp = [], [], []
    for ds in MAIN:
        t = [unit(ds, s, "ind05", "v2")["prior_test"][unit(ds, s, "ind05", "v2")["minority_class"]]
             for s in SEEDS]
        a = [unit(ds, s, "ind05", "v2")["methods"]["tabpfn"]["predicted_minority_rate"] for s in SEEDS]
        b = [unit(ds, s, "ind05", "v2")["methods"]["picl"]["predicted_minority_rate"] for s in SEEDS]
        tx.append(st.fmean(t)); pt.append(st.fmean(a)); pp.append(st.fmean(b))
    fig, ax = plt.subplots(figsize=(3.35, 2.75))
    ax.plot([0, 0.46], [0, 0.46], color="0.60", linewidth=0.7, linestyle="--", zorder=0)
    ax.scatter(tx, pt, s=24, color=GREY, zorder=3, label="TabPFN")
    ax.scatter(tx, pp, s=24, color=BLUE, zorder=3, label="PICL-Prior")
    ax.set_xlabel("Minority rate in the test fold", fontsize=8)
    ax.set_ylabel("Predicted minority rate", fontsize=8)
    ax.set_xlim(0, 0.46); ax.set_ylim(0, 0.46)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=6.5, frameon=False, loc="upper left")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "fig3_prior.pdf"), bbox_inches="tight")
    plt.close(fig)


def main_ga():
    """Graphical abstract: scatter (left) and induced-5% bars (right), 1328x531 px."""
    xs, tab, picl = [], [], []
    for ds in MAIN:
        au = [unit(ds, s, "ind05", "v2")["methods"]["tabpfn"]["auroc"] for s in SEEDS]
        ft = [unit(ds, s, "ind05", "v2")["methods"]["tabpfn"]["macro_f1"] for s in SEEDS]
        fp = [unit(ds, s, "ind05", "v2")["methods"]["picl"]["macro_f1"] for s in SEEDS]
        xs.append(st.fmean(au)); tab.append(st.fmean(ft)); picl.append(st.fmean(fp))
    order = ["TabPFN", "+OS", "BalCtx", "Thr", "PICL", "LogReg", "XGB-w"]
    key = {"TabPFN": "tabpfn", "+OS": "os", "BalCtx": "balctx", "Thr": "thr",
           "PICL": "picl", "LogReg": "logreg", "XGB-w": "xgboost_w"}
    g = load(SUMMARY)["grids"]["main12|ind05|v2"]["methods"]
    vals = [g[key[m]]["mean"] for m in order]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.28, 5.31), dpi=100,
                                   gridspec_kw={"width_ratios": [1, 1.05]})
    for x, a, b in zip(xs, tab, picl):
        ax1.plot([x, x], [a, b], color="0.80", linewidth=1.0, zorder=1)
    ax1.plot([0.68, 1.0], [0.68, 1.0], color="0.60", linewidth=0.8, linestyle="--", zorder=0)
    ax1.scatter(xs, tab, s=42, color=GREY, zorder=3, label="TabPFN argmax")
    ax1.scatter(xs, picl, s=42, color=BLUE, zorder=3, label="Prior-corrected")
    ax1.set_xlabel("AUROC", fontsize=13); ax1.set_ylabel("Macro-F1", fontsize=13)
    ax1.set_xlim(0.70, 1.01); ax1.set_ylim(0.33, 1.0)
    ax1.set_title("Induced 5%: ranking holds; decisions collapse", fontsize=14)
    ax1.tick_params(labelsize=11)
    ax1.legend(fontsize=11.5, frameon=False, loc="lower right")
    for s in ("top", "right"):
        ax1.spines[s].set_visible(False)

    cols = [METHOD_COLOR[m] for m in order]
    ax2.bar(range(len(order)), vals, color=cols, zorder=3)
    ax2.set_xticks(range(len(order)))
    ax2.set_xticklabels(order, fontsize=12, rotation=28, ha="right")
    ax2.set_ylabel("Mean macro-F1", fontsize=13)
    ax2.set_ylim(0, 0.86)
    ax2.set_title("Induced 5%: do not resample the prompt", fontsize=14)
    ax2.tick_params(labelsize=11)
    for s in ("top", "right"):
        ax2.spines[s].set_visible(False)

    fig.suptitle("Calibrate before use for tabular in-context learning", fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    os.makedirs(GA, exist_ok=True)
    fig.savefig(os.path.join(GA, "graphical_abstract.pdf"))
    fig.savefig(os.path.join(GA, "graphical_abstract.png"), dpi=100)
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(FIGS, exist_ok=True)
    main_scatter(); print("fig1_rank.pdf")
    main_bars(); print("fig2_bars.pdf")
    main_prior(); print("fig3_prior.pdf")
    main_ga(); print("graphical_abstract.pdf/.png")
