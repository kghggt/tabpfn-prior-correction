# TabPFN prior correction — code and frozen results

Companion repository for the manuscript

> **Calibrate Before Use for Tables: The Majority-Label Bias of Tabular In-Context Learning**
> Lingfeng Wang, Ming Li, Jianlong Chang — *Pattern Recognition Letters* (under review)

Every number, table and figure in the letter comes from the artefacts in this repository.
No cell in the paper's tables was typed by hand: the LaTeX tables are generated from `results/`
by `code/export_tables.py`, and the plain-text list that the letter quotes is
`results/report_numbers.md`.

---

## Layout

| Path | What it is |
|---|---|
| `code/` | the full pipeline: data loading and context budgeting, the 19 compared methods, the run drivers, the analysis (paired Wilcoxon tests, cluster merging, collapse/fidelity strata), LaTeX table export, and the plain-text number list |
| `results/units/` | **one JSON per run unit (414 files)**; each holds the per-dataset metrics of every method for a single (dataset, seed, setting, checkpoint) |
| `results/summary.json` | grid-level aggregation and paired tests (27 grids) |
| `results/report_numbers.md` | the single traceable source for every number quoted in the letter |
| `results/detector.json` | deployable-fidelity strata (train-side hold-out diagnostic) |
| `results/shift_probe.json` | subpopulation-shift probe (naturally occurring prior gaps) |
| `results/ext_probe.json` | extended-grid candidate screening from OpenML metadata, recorded before the protocol was frozen |
| `results/bench_mem.json` | GPU memory / wall-clock benchmark behind the context-budget rule |
| `results/openml_index.json` | cached OpenML metadata (dataset ids, sizes, class counts) |
| `figures/` | the published figures: Fig. 1–4 and the graphical abstract |

## What produced what

| Paper element | Produced by |
|---|---|
| Table 1 and supplementary Tables S1–S9 | `code/export_tables.py` (reads `results/summary.json`) |
| Every number quoted in the text | `results/report_numbers.md`, written by `code/report_numbers.py` |
| Figures 1–3 and the graphical abstract | `code/plot_figures.py` |
| Figure 4 | `code/plot_rule.py` |
| Twelve-table main grid (natural / induced 10 % / induced 5 %, seeds 0–2) | `code/run_main.py --suite main12` |
| Naturally rare panel (12 tables, no downsampling) | `code/run_main.py --suite rare12` |
| Extended grid (18 tables, unused when the protocol was chosen) | `code/run_main.py --suite extended` |
| Subpopulation shift (no label resampled) | `code/run_shift.py` |
| Deployable fidelity diagnostic | `code/run_diag.py`, `code/detector.py` |
| Aggregation, paired tests, strata | `code/analyze.py` |

## Reproducing

Requires Python ≥ 3.10, a CUDA GPU is strongly recommended (all runs in the letter used a 6 GB
RTX 3060), and:

```bash
pip install "tabpfn==9.0.0" torch scikit-learn pandas numpy matplotlib catboost xgboost lightgbm
```

Fetch and pin the two TabPFN checkpoints used in the letter (the code always loads them from local
files, so a weight swap is the only difference between the two checkpoint generations):

```bash
python code/fetch_models.py --versions v2 v2_5
```

Then run the grids, aggregate, and export:

```bash
python code/run_main.py --suite smoke     # quick sanity check, two datasets
python code/run_main.py --suite main12    # twelve-table grid
python code/run_main.py --suite rare12    # naturally rare panel
python code/run_main.py --suite extended  # eighteen further tables
python code/run_shift.py                  # subpopulation-shift protocol
python code/run_diag.py                   # train-side fidelity diagnostic
python code/analyze.py                    # -> results/summary.json
python code/export_tables.py              # -> LaTeX tables
python code/report_numbers.py             # -> results/report_numbers.md
python code/plot_figures.py               # -> figures/fig1_rank, fig2_bars, fig3_prior, graphical_abstract
python code/plot_rule.py                  # -> figures/fig4_rule.pdf
```

Runs are resumable: each unit writes one JSON into `results/units/` and is skipped if it already
exists, so a long grid can be interrupted and restarted. Context length is bounded by a
*cell budget* (`(n_ctx + n_test) × n_test`) rather than a row count, applied identically to every
method inside a unit so that comparisons stay paired; the realised sizes are recorded in each
artefact.

`code/_probe_ext.py` screens the extended-grid candidates by OpenML metadata (two classes, 300–12000
rows, minority share ≥ 2 %, loadable) before the protocol is frozen. `code/_probe.py`,
`code/_bench_mem.py`, `code/_dl.py` and `code/run_strengthen.sh` are auxiliary diagnostics and the
original end-to-end driver; files prefixed with `_` are utilities rather than part of the method.

### A note on paths

`code/config.py` resolves `results/`, `figures/` and `paper/tables/` relative to the repository root,
and the two plotting scripts write into `figures/`. The plotting scripts are the ones used for the
letter with their output paths re-pointed at this repository's layout; everything else is unchanged.

## License

- Code (`code/`): MIT — see [LICENSE](LICENSE).
- Frozen results and figures (`results/`, `figures/`): CC BY 4.0.

## Data

All datasets are public OpenML tables; `results/openml_index.json` and the `openml_id` field inside
each unit record the exact source of every table. No new data were collected.
