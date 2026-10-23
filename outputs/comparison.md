# Method comparison (validation: train weeks 11–20, truth weeks 21–30)

Cohort A = 1150 submission users, cohort C = 4030 log users (each with ≥ 5 programmes in the truth). All variants are listed, including the worst. `selected` marks the variant each family selected on MAP@5 A.

Columns: **hit_A / hit_C** = share of users with ≥ 1 of the 5 ids in their truth; **repeats_A** = share of recommended slots already watched by the user in weeks 11–20; **mean_popularity_A** = mean number of distinct users (of 4737 with a watched airing in weeks 11–20) who watched the recommended programme in weeks 11–20; **distinct_ids_A** = distinct programmes recommended across cohort A; **runtime_s** — baseline: ranking + fill for 4,030 users; ppr: PageRank iterations + ranking for 4,030 users; graph build is shared and listed once; mf: ALS fit + scoring for 4,030 users; hybrid: PPR graph build + ranking + MF fit/ranking (depth 50) + fusion (PPR graph build: 0.09 s).

## All methods

| family | variant | map@5_A | map@5_C | hit_A | hit_C | repeats_A | mean_popularity_A | distinct_ids_A | runtime_s | selected |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline | popularity | 0.235699 | 0.213795 | 0.773 | 0.7367 | 0.6777 | 2836.8 | 5 | 0.05 | yes |
| baseline | personal_history | 0.343832 | 0.30421 | 0.8704 | 0.8181 | 0.9798 | 1677.8 | 492 | 0.17 | yes |
| ppr | alpha=0.1 | 0.353838 | 0.31627 | 0.8826 | 0.8427 | 0.9798 | 1895.0 | 258 | 1.01 |  |
| ppr | alpha=0.2 | 0.354325 | 0.31687 | 0.8835 | 0.8437 | 0.9793 | 1915.0 | 255 | 1.12 |  |
| ppr | alpha=0.3 | 0.354913 | 0.317993 | 0.8835 | 0.8447 | 0.9779 | 1962.6 | 246 | 1.25 | yes |
| ppr | alpha=0.5 | 0.354304 | 0.31872 | 0.8826 | 0.8469 | 0.9631 | 2139.7 | 216 | 1.53 |  |
| ppr | alpha=0.7 | 0.338701 | 0.305031 | 0.8791 | 0.8471 | 0.8929 | 2455.0 | 153 | 1.66 |  |
| ppr | alpha=0.85 | 0.305641 | 0.275404 | 0.8652 | 0.8268 | 0.7843 | 2699.2 | 66 | 1.93 |  |
| mf | f64_reg0.01_a10 | 0.167461 | 0.161546 | 0.7478 | 0.7132 | 0.8217 | 1541.5 | 228 | 1.35 | yes |
| mf | f64_reg0.01_a40 | 0.101223 | 0.105854 | 0.533 | 0.5454 | 0.555 | 1206.7 | 376 | 1.3 |  |
| mf | f64_reg0.1_a10 | 0.166736 | 0.160682 | 0.7348 | 0.7057 | 0.8283 | 1534.5 | 233 | 1.3 |  |
| mf | f64_reg0.1_a40 | 0.106058 | 0.109177 | 0.5609 | 0.5591 | 0.5826 | 1207.3 | 390 | 1.31 |  |
| hybrid | w=0 | 0.167461 | 0.161546 | 0.7478 | 0.7132 | 0.8217 | 1541.5 | 228 | 2.38 |  |
| hybrid | w=0.25 | 0.232206 | 0.214816 | 0.8374 | 0.7931 | 0.9296 | 1719.2 | 189 | 2.37 |  |
| hybrid | w=0.5 | 0.308638 | 0.278251 | 0.8826 | 0.8357 | 0.9628 | 1895.6 | 190 | 2.38 |  |
| hybrid | w=0.75 | 0.346348 | 0.311831 | 0.8817 | 0.8412 | 0.9739 | 1945.7 | 213 | 2.37 |  |
| hybrid | w=1 | 0.354913 | 0.317993 | 0.8835 | 0.8447 | 0.9779 | 1962.6 | 246 | 2.38 | yes |


## MF diagnostics — off-grid, not used for selection (вне сетки, для выбора не использовались)

| family | variant | map@5_A | map@5_C | hit_A | hit_C | repeats_A | mean_popularity_A | distinct_ids_A | runtime_s | selected |
|---|---|---|---|---|---|---|---|---|---|---|
| mf_off_grid | counts_f64_reg0.01_a1 | 0.178954 | 0.171451 | 0.7539 | 0.7387 | 0.9565 | 1659.7 | 121 | 1.24 |  |
| mf_off_grid | binary_f64_reg0.01_a10 | 0.185087 | 0.177158 | 0.7791 | 0.7469 | 0.8628 | 1642.5 | 202 | 1.3 |  |
| mf_off_grid | log1p_f64_reg0.01_a10 | 0.191238 | 0.181744 | 0.7913 | 0.7506 | 0.8957 | 1655.8 | 180 | 1.31 |  |


## Checks

- Hybrid: {"w=0 equals MF": {"users_identical": 4030, "users": 4030}, "w=1 equals PPR": {"users_identical": 4030, "users": 4030}}

- MF reproducibility (best config fitted twice): {"user_factors_identical": true, "item_factors_identical": true, "max_abs_factor_diff": 0.0, "predictions_identical": true}

- MAP@5 recomputed here from the saved prediction files equals the stored value for every row.
