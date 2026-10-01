"""PRL 补强实验的统一配置。

约定（产物隔离）：
  * 本文件及 code/ 下所有脚本只**读**基线稿件目录 restored/，绝不写入。
  * 一切运行产物写入 results/、logs/、paper/tables/（自动生成）。
  * 所有数字必须来自真实运行，禁止手写或回填。
"""

from __future__ import annotations

import os

# ---------------------------------------------------------------- 路径隔离

VENUE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CODE_DIR = os.path.join(VENUE_ROOT, "code")
RESULTS_DIR = os.path.join(VENUE_ROOT, "results")
LOGS_DIR = os.path.join(VENUE_ROOT, "logs")
PAPER_DIR = os.path.join(VENUE_ROOT, "paper")
PAPER_TABLES_DIR = os.path.join(PAPER_DIR, "tables")
FIGURES_DIR = os.path.join(PAPER_DIR, "figures")

# 基线稿件目录：只读，任何脚本不得写入
BASELINE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(VENUE_ROOT), "..", "restored")
)

for _d in (RESULTS_DIR, LOGS_DIR, PAPER_TABLES_DIR, FIGURES_DIR):
    os.makedirs(_d, exist_ok=True)

# ---------------------------------------------------------------- 实验设定

# 主网格：论文原 twelve-table grid
MAIN12 = [
    "australian",
    "blood-transfusion",
    "credit-g",
    "diabetes",
    "ilpd",
    "kc1",
    "pc1",
    "phoneme",
    "qsar-biodeg",
    "spambase",
    "wdbc",
    "wilt",
]

# 自然稀有面板：论文原 rare panel（minority <= 15.5%，不做训练下采样）
RARE12 = [
    "mammography",
    "mc1",
    "wilt",
    "sick",
    "ozone-8hr",
    "pc1",
    "mw1",
    "climate-model",
    "kc3",
    "pc3",
    "pc4",
    "kc1",
]

# 扩展网格：为提升统计功效新增的 OpenML 二分类表。
# 名单由 code/_probe_ext.py 用 OpenML 元数据筛出（二分类、300..12000 行、少数类占比 >= 2%），
# 探测结果在 results/ext_probe.json，不做手工增删。
EXTENDED = [
    "kr-vs-kp",
    "churn",
    "SpeedDating",
    "Titanic",
    "ionosphere",
    "breast-w",
    "colic",
    "credit-approval",
    "banknote-authentication",
    "cylinder-bands",
    "hill-valley",
    "ozone-level-8hr",
    "thoracic-surgery",
    "tic-tac-toe",
    "monks-problems-1",
    "monks-problems-2",
    "tokyo1",
    "sylvine",
]

SEEDS_EXTENDED = [0]

# rare panel 中的同源簇：NASA/PROMISE 软件缺陷表彼此高度同源，
# 逐表当作独立样本会高估统计功效，检验时把这些表先平均成一个观测。
RARE_CLUSTERS = {
    "kc1": "nasa_promise",
    "kc3": "nasa_promise",
    "pc1": "nasa_promise",
    "pc3": "nasa_promise",
    "pc4": "nasa_promise",
    "mw1": "nasa_promise",
    "mc1": "nasa_promise",
}

# 天然先验偏移场景：不重采样，训练/测试取自患病率不同的两个自然子群
SHIFT_DATASETS = [
    "adult",
    "bank-marketing",
    "default-of-credit-card-clients",
    "credit-g",
    "Titanic",
    "churn",
    "kddcup09_appetency",
    "KDDCup09-Upselling",
    "electricity",
    "occupancy-detection",
    "blood-transfusion",
    "SpeedDating",
]

# OpenML 上名称与常用简称不一致的数据集，直接钉死 id 以保证可复现
DATASET_ID_OVERRIDE = {
    "blood-transfusion": 1464,  # OpenML 上名为 "Blood transfusion service center"
    "ozone-8hr": 1487,  # OpenML 上名为 "ozone-level-8hr"（UCI 8 小时臭氧浓度）
    "climate-model": 1467,  # OpenML 上名为 "climate-model-simulation-crashes"
}

# 训练先验设定：natural 不改标签；induced 下采样少数类
SETTINGS = ["natural", "ind05", "ind10"]
INDUCED_RATIO = {"ind05": 0.05, "ind10": 0.10}
MIN_MINORITY_ROWS = 5  # 下采样后少数类至少保留的行数

SEEDS_MAIN = [0, 1, 2]
SEEDS_RARE = [0, 1, 2, 3, 4]

# TabPFN 模型版本：v2 对应论文原 tabpfn==2.2.1 的复现对照；v2_5 / v3_5 为版本升级
MODEL_VERSIONS = ["v2", "v2_5"]
MODEL_VERSIONS_OPTIONAL = ["v3_5"]

# HuggingFace 仓库与默认权重文件名，取自 tabpfn 9.0.0 的 ModelSource 定义。
# 注意：TabPFNClassifier(model_path="v2") 会把 "v2" 当成**文件名**而非版本，
# 因此这里统一走"先下载到本地、再传绝对路径"，既规避该坑也保证产物隔离。
MODEL_SOURCES = {
    "v2": ("Prior-Labs/TabPFN-v2-clf", "tabpfn-v2-classifier-finetuned-zk73skhh.ckpt"),
    "v2_5": ("Prior-Labs/tabpfn_2_5", "tabpfn-v2.5-classifier-v2.5_default.ckpt"),
    "v2_6": ("Prior-Labs/tabpfn_2_6", "tabpfn-v2.6-classifier-v2.6_default.ckpt"),
    "v3": ("Prior-Labs/tabpfn_3", "tabpfn-v3-classifier-v3_default.ckpt"),
    "v3_5": ("Prior-Labs/tabpfn_3_5", "tabpfn-v3.5-20260909.safetensors"),
    "v3_5_fast": ("Prior-Labs/tabpfn_3_5", "tabpfn-v3.5-fast-20260909.safetensors"),
}

# 权重本地缓存目录（工作区内，避免写入被沙箱拦截的系统目录）
MODELS_DIR = os.path.join(VENUE_ROOT, ".cache", "models")
os.makedirs(MODELS_DIR, exist_ok=True)


def model_path(version: str) -> str:
    """返回本地权重绝对路径；不存在则抛错，要求先跑 fetch_models.py。"""
    if version not in MODEL_SOURCES:
        raise KeyError(f"未知模型版本: {version}")
    _, filename = MODEL_SOURCES[version]
    p = os.path.join(MODELS_DIR, filename)
    if not os.path.exists(p):
        raise FileNotFoundError(f"权重缺失，请先运行: python code/fetch_models.py --versions {version}")
    return p


def available_versions() -> list:
    return [v for v, (_, f) in MODEL_SOURCES.items() if os.path.exists(os.path.join(MODELS_DIR, f))]

# 上下文预算：超过则随机子采样训练侧。
# 可用环境变量 TABPFN_MAX_CONTEXT 覆盖，用于上下文预算消融（不改代码即可换档）。
MAX_CONTEXT_ROWS = int(os.environ.get("TABPFN_MAX_CONTEXT", "4096"))

# 测试集行数上限。可用 TABPFN_MAX_TEST 覆盖。
MAX_TEST_ROWS = int(os.environ.get("TABPFN_MAX_TEST", "1000"))

# ---------------------------------------------------------------- 显存预算
#
# 单次前向的代价大致正比于 cells = (n_train + n_test) * n_test，并随特征维度上升。
# 在 6GB 的 RTX 3060 上实测（code/_bench_mem.py，产物 results/bench_mem.json）：
#   d=20 : 3.05M cells -> 3.66 GB / 3.6 s    5.10M cells -> 7.19 GB / 37.6 s
#   d=57 : 0.77M cells -> 2.78 GB / 2.9 s    2.36M cells -> 10.6 GB / 118 s
# 超过物理显存时 Windows 会用共享内存换页，"跑得完"但慢一到两个数量级，
# 而且第一次真 OOM 之后 CUDA 会异步报错，把后面一连串小表全部拖死。
# 所以这里按维度给一个 cells 预算，再据此同时决定测试侧与上下文侧上限，
# 而不是拍脑袋只卡其中一边。
CELL_BUDGET_BUCKETS = (
    (24, 3_000_000),
    (48, 1_600_000),
    (96, 800_000),
)
CELL_BUDGET_WIDE = 500_000  # d > 96
MIN_TEST_ROWS = 256


# 全局预算缩放：某个单元仍 OOM 时，run_grid 会把它在**该单元内**折半重试。
BUDGET_SCALE = float(os.environ.get("TABPFN_BUDGET_SCALE", "1.0"))


def cell_budget(n_features: int | None) -> int:
    d = int(n_features or 8)
    for hi, budget in CELL_BUDGET_BUCKETS:
        if d <= hi:
            return int(budget * BUDGET_SCALE)
    return int(CELL_BUDGET_WIDE * BUDGET_SCALE)


def fit_caps(n_test: int, n_features: int | None) -> tuple[int, int]:
    """按 cells 预算返回 (测试侧上限, 上下文侧上限)。

    两边同时让位：测试侧取 min(MAX_TEST_ROWS, sqrt(budget/2))，
    上下文侧取 budget/n_test - n_test，再夹到 [256, MAX_CONTEXT_ROWS]。
    """
    budget = cell_budget(n_features)
    te_cap = min(MAX_TEST_ROWS, int((budget / 2.0) ** 0.5))
    n_te = max(MIN_TEST_ROWS, min(int(n_test), te_cap))
    ctx_cap = int(budget / max(n_te, 1)) - n_te
    ctx_cap = max(256, min(MAX_CONTEXT_ROWS, ctx_cap))
    return n_te, ctx_cap

# 主网格方法名（与表格列一致）
PROMPT_METHODS = ["tabpfn", "os", "smote", "balctx"]
POSTHOC_METHODS = [
    "picl",  # 目标先验 = 均匀（论文原 PICL-Prior）
    "distpfn",  # 目标先验 = 测试批次平均预测分布（Lee 2026 官方实现）
    "distpfn_t",  # 同上 + 温度缩放
    "thr",  # 冻结分数 + hold-out 阈值
    "picl_full",  # 温度缩放 + 先验校正
    "picl_oracle",  # 目标先验 = 测试集真实患病率（非可部署，仅诊断）
]
NON_TABPFN_BASELINES = ["logreg", "xgboost", "catboost", "lightgbm"]

# ---------------------------------------------------------------- 运行参数

# 与论文保持一致；可用 TABPFN_N_EST 覆盖以跑 estimator 数量消融。
N_ESTIMATORS = int(os.environ.get("TABPFN_N_EST", "4"))
IGNORE_PRETRAINING_LIMITS = True
HOLDOUT_FRACTION = 0.20  # 阈值/温度搜索用的训练留出比例
SMOTE_K = 5
RANDOM_STATE = 0

# 单个数据集的最长等待（秒）；超时则跳过并记录
PER_DATASET_TIMEOUT = 900

# prompt 级方法（+OS / +SMOTE）会把训练集放大到多数类的两倍，
# 最大的表上会超过万行。超过该硬上限时做分层子采样，避免显存耗尽后
# 落到 CPU 前向（那会慢到不可接受）。
# 默认与上下文预算一致：任何 prompt（含过采样放大后的）都不超过 4096 行。
# 这对 +OS/+SMOTE 的语义无损（分层子采样后仍是类平衡的重复/插值样本），
# 但不这样做，最大那几张表的过采样 prompt 会到上万行，单次前向就要好几分钟。
HARD_CONTEXT_CAP = int(os.environ.get("TABPFN_HARD_CONTEXT_CAP", str(MAX_CONTEXT_ROWS)))

# CPU 兜底默认关闭：开启后单个大表可能跑数小时，掩盖失败。
ALLOW_CPU_FALLBACK = os.environ.get("TABPFN_ALLOW_CPU", "0") == "1"

# 运行变体标签：非空时写进单元文件名与 grid 字段，用于消融实验与主网格隔离。
# 例：TABPFN_N_EST=8 TABPFN_RUN_TAG=est8 跑 8 estimators 的消融。
RUN_TAG = os.environ.get("TABPFN_RUN_TAG", "")
