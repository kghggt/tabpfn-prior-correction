"""重画正文 Figure 4：uniform 校正带来的 macro-F1 变化 vs 保真度比值。

横轴 ratio = vanilla TabPFN 预测的少数类率 / 测试折少数类率（先对种子取均值，
再逐数据集一个点）；纵轴 = PICL(uniform 目标) 的 macro-F1 减去 vanilla TabPFN。
两个面板：主网格 induced 5% 与自然稀有面板，均为 v2 权重。
所有数字来自 results/units/*.json，无手写值。
"""

from __future__ import annotations

import glob
import json
import os
import statistics as st
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import config  # noqa: E402

PANELS = [
    ("Induced 5\\%", "main12", "ind05", "v2", "o", "#d62728"),
    ("Natural rare", "rare12", "natural", "v2", "s", "#17557e"),
]

ANNOTATE = {"diabetes", "climate-model", "sick", "wilt", "mammography", "mc1"}

OUT = os.path.join(config.VENUE_ROOT, "figures", "fig4_rule.pdf")


def load_units() -> list[dict]:
    units = []
    for path in sorted(glob.glob(os.path.join(config.RESULTS_DIR, "units", "*.json"))):
        with open(path, encoding="utf-8") as fh:
            units.append(json.load(fh))
    return units


def per_dataset(units, grid, setting, version):
    agg = defaultdict(lambda: {"ratio": [], "tab": [], "picl": []})
    for u in units:
        if u.get("grid", "main12") != grid or u["setting"] != setting:
            continue
        if u["model_version"] != version:
            continue
        pt = u["prior_test"]
        r_test = pt[u["minority_class"]] if isinstance(pt, list) else pt
        if not r_test:
            continue
        a = agg[u["dataset"]]
        a["ratio"].append(u["methods"]["tabpfn"]["predicted_minority_rate"] / r_test)
        a["tab"].append(u["methods"]["tabpfn"]["macro_f1"])
        a["picl"].append(u["methods"]["picl"]["macro_f1"])
    out = {}
    for ds, a in agg.items():
        out[ds] = (
            st.fmean(a["ratio"]),
            st.fmean(a["picl"]) - st.fmean(a["tab"]),
        )
    return out


def main() -> int:
    units = load_units()
    labelled: set[str] = set()
    fig, ax = plt.subplots(figsize=(3.4, 2.7))
    for label, grid, setting, version, marker, color in PANELS:
        pts = per_dataset(units, grid, setting, version)
        xs = [v[0] for v in pts.values()]
        ys = [v[1] for v in pts.values()]
        ax.scatter(xs, ys, s=22, marker=marker, color=color, label=label.replace("\\%", "%"), zorder=3)
        for ds, (x, y) in sorted(pts.items(), key=lambda kv: -kv[1][1]):
            # 同名表在两个面板都出现（如 wilt）时只标一次，避免标签叠字
            if ds in ANNOTATE and ds not in labelled:
                labelled.add(ds)
                ax.annotate(
                    ds, (x, y), textcoords="offset points", xytext=(5, 3),
                    fontsize=6, color=color, zorder=4,
                )
        print(f"{label}: n={len(pts)}")
        for ds, (x, y) in sorted(pts.items()):
            print(f"   {ds:<16} ratio {x:.2f}  Δ {y:+.3f}")

    ax.axhline(0.0, color="0.6", linewidth=0.7, linestyle="--", zorder=1)
    ax.set_xlabel("Predicted minority rate / test minority rate", fontsize=8)
    ax.set_ylabel("Macro-F1, PICL $-$ TabPFN", fontsize=8)
    ax.set_xlim(-0.03, 1.15)
    ax.set_ylim(-0.22, 0.40)
    ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=6.5, frameon=False, loc="upper right")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight")
    print("写出:", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
