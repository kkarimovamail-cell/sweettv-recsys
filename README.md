# sweet.tv — TV programme recommender

Course project on the Kaggle community competition
[sweet.tv — TV program recommender](https://www.kaggle.com/competitions/sweettv-tv-program-recommender):
for each of 1,260 users, predict the 5 TV programmes they will watch in ISO weeks 31–42 of 2020, scored by MAP@5.

The pipeline joins the viewing log to the broadcast schedule, evaluates several recommenders on one temporal hold-out
and ends with a learned re-ranker on top of Personalized PageRank and channel × time-slot habits.

## Final submission

| | |
|---|---|
| File | [`submissions/submission_v7.csv`](submissions/submission_v7.csv) |
| md5 | `d0e51925c3911a9174817c3de129e94c` |
| Method | v7 — `LGBMRanker` (lambdarank) re-ranking on top of v6 (PPR + airing prior + recency + soft weights + slot habits) |
| **Kaggle private MAP@5 (graded score)** | **0.43671** |
| Kaggle public MAP@5 (reference only) | 0.43802 |

Only the best Kaggle **private** score is graded; public scores are given for reference. Kaggle scores were reported
by the author from the Kaggle submission page; the pipeline itself has no Kaggle access.

## Results

Validation: train on ISO weeks 11–20, ground truth = each user's top-5 programmes by watched airings
(screen_time ≥ 0.8) in weeks 21–30. Cohort A = 1,150 submission users with ≥ 5 truth programmes, cohort C = all
4,030 such users of the log. v7 is evaluated out-of-fold (two folds of cohort-C users).

| Method | MAP@5 A | MAP@5 C | Kaggle private | Kaggle public (ref.) |
|---|---|---|---|---|
| Popularity (baseline) | 0.2357 | 0.2138 | — | — |
| Personal history (reference) | 0.3438 | 0.3042 | — | — |
| Matrix factorization (implicit ALS) | 0.1675 | 0.1615 | — | — |
| Content-based filtering | 0.2071 | 0.1824 | — | — |
| v1 — Personalized PageRank, α 0.3 | 0.3549 | 0.3180 | 0.39806 | 0.40746 |
| v2 — + airing prior (β 0.25) + recency (H 4 weeks) | 0.3667 | 0.3294 | 0.39944 | 0.40299 |
| v5 — + soft training weights (screen_time ≥ 0.1) | 0.3776 | 0.3416 | 0.41627 | 0.42955 |
| v6 — + channel × day × hour habits (λ 2) | 0.3835 | 0.3466 | 0.42440 | 0.43613 |
| **v7 — + LightGBM re-ranking (final)** | **0.3936** | **0.3543** | **0.43671** | 0.43802 |

Details: [`outputs/comparison.md`](outputs/comparison.md), [`outputs/design_choices.md`](outputs/design_choices.md),
[`outputs/two_users.md`](outputs/two_users.md), [`outputs/issues_log.md`](outputs/issues_log.md).

## Data

The CSV files are **not** in this repository. Accept the competition rules on Kaggle, download the data and put the
five files into a folder named `sweettv-tv-program-recommender` **next to** the repository folder
(`src/config.py`: `DATA_DIR = PROJECT_ROOT.parent / "sweettv-tv-program-recommender"`):

```
parent/
├── <this repository>/
└── sweettv-tv-program-recommender/
    ├── dataset11-30.csv
    ├── export_arh_11-20-final.csv
    ├── export_arh_21-30-final.csv
    ├── export_arh_31-42-final.csv
    └── submission.csv
```

The raw files are only read, never modified.

## Installation

Python 3.12 (developed on 3.12.7, macOS arm64).

```bash
python3.12 -m venv .venv
```

```bash
.venv/bin/python -m pip install -r requirements.txt
```

## Running

Quick check of the whole project — the notebook rebuilds everything from the raw CSV files (about 1.5 minutes on an
Apple M4), writes only to `outputs/nb_check/` and asserts that its numbers match `outputs/*.json` and that the rebuilt
final file has the md5 above:

```bash
.venv/bin/jupyter nbconvert --to notebook --execute --inplace notebooks/sweettv_recsys.ipynb
```

Unit tests:

```bash
.venv/bin/pytest
```

The step-by-step scripts (each writes its results into `outputs/`) in the order they were run:
`src.audit` → `src.join` → `src.protocol` → `src.baselines` → `src.ppr` → `src.mf` → `src.hybrid` →
`src.comparison` → `src.improve` → `src.expected` → `src.cbf` → `src.signal_exp` → `src.slots` → `src.rerank` →
`src.rerank_v8` → `src.two_users`, each as `.venv/bin/python -m src.<name>` from the repository root.
`src.final` produced the first final (v1); `outputs/results_final.json` was afterwards updated with v7 and the Kaggle
scores, so re-running `src.final` would overwrite it.

## Repository structure

| Path | Contents |
|---|---|
| [`src/config.py`](src/config.py) | Paths, seed 42, thresholds, logger |
| [`src/audit.py`](src/audit.py) | Data audit: row counts, date trap, time-zone test, parquet cache |
| [`src/join.py`](src/join.py) | Sessions → (user, airing) interactions with screen_time |
| [`src/protocol.py`](src/protocol.py), [`src/metrics.py`](src/metrics.py) | Validation protocol, cohorts, pools, MAP@5 |
| [`src/baselines.py`](src/baselines.py) | Popularity and personal-history baselines |
| [`src/ppr.py`](src/ppr.py), [`src/mf.py`](src/mf.py), [`src/cbf.py`](src/cbf.py) | Personalized PageRank, implicit ALS, content-based filtering |
| [`src/hybrid.py`](src/hybrid.py), [`src/improve.py`](src/improve.py), [`src/expected.py`](src/expected.py), [`src/signal_exp.py`](src/signal_exp.py), [`src/slots.py`](src/slots.py) | Hybrids and improvement steps v2–v6 |
| [`src/rerank.py`](src/rerank.py), [`src/rerank_v8.py`](src/rerank_v8.py) | LightGBM re-ranking (v7) and its refinements |
| [`src/submission.py`](src/submission.py) | Writing and validating submission files |
| [`src/two_users.py`](src/two_users.py), [`src/comparison.py`](src/comparison.py), [`src/final.py`](src/final.py) | Report tables and the first final |
| [`tests/`](tests/) | Unit tests with hand-computed expectations |
| [`notebooks/sweettv_recsys.ipynb`](notebooks/sweettv_recsys.ipynb) | Executed end-to-end notebook |
| [`outputs/results_validation.json`](outputs/results_validation.json) | Every validation number |
| [`outputs/results_final.json`](outputs/results_final.json) | Final configuration, md5, Kaggle submissions |
| [`outputs/data_audit.json`](outputs/data_audit.json), [`outputs/join_stats.json`](outputs/join_stats.json) | Data audit and join statistics |
| [`outputs/issues_log.md`](outputs/issues_log.md) | Every error and data problem: symptom, cause, fix, numbers before/after |
| `outputs/logs/` | Run logs |
| [`submissions/submission_v7.csv`](submissions/submission_v7.csv) | Final submission |

## Reproducibility

Seed 42 everywhere; ALS with one thread and `random_state=42`; LightGBM with `deterministic=True`, `n_jobs=1`,
`random_state=42`. Exact package versions are in [`requirements.txt`](requirements.txt). The notebook's last section
prints the environment and per-section runtimes.

Developed with AI assistance (Claude Code); the full session export accompanies the report.
