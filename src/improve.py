"""One-variable improvements of the final method (PPR, alpha from validation), validation only.

  exp 1  airing prior:  score = PPR mass * n_airings ** beta, n_airings = labelled airings of the programme
                        in the target window (validation: ISO weeks 21-30, test: 31-42; items.parquet).
                        Uses the schedule only (which programme airs when), never target-window viewing.
  exp 2  recency:       each watched airing weighs 0.5 ** (delta / H), delta = end week of the train window
                        minus the airing's ISO week; applied to the restart (sum of weights) and to the graph
                        (weight of the most recent watched airing). H = inf is the current PPR.
  exp 3  combination:   only if both experiments beat the baseline: neighbours of the best beta x best H.
Selection on MAP@5 A; a variant replaces the baseline only if strictly better.

Run from the project root:  python -m src.improve
"""
import hashlib
import json
import math
import time

import numpy as np
import pandas as pd

from src import config, ppr
from src import protocol as P
from src.hybrid import load_preds
from src.metrics import apk
from src.submission import validate_submission, write_submission

BETAS = [0.0, 0.25, 0.5, 0.75, 1.0]
HALF_LIVES = [math.inf, 8.0, 4.0, 2.0]  # inf goes through the decay formula: 0.5 ** (delta / inf) = 1
SELECT_ON = f"map@{P.K}_A"

log = config.get_logger("11_improve")


def h_label(h) -> str:
    return "inf" if math.isinf(h) else f"{h:g}"


def airing_weight(items_df: pd.DataFrame, graph_items: np.ndarray, split: str, beta: float) -> np.ndarray:
    n = (items_df.set_index("tv_show_id")[P.SPLITS[split]["pool_col"]]
         .reindex(graph_items, fill_value=0).to_numpy(dtype=float))
    return np.power(n, beta)  # pool programmes all have n >= 1, so beta = 0 gives exactly 1


def ap_change(preds: dict, base: dict, truth: dict, users: list[str]) -> dict:
    diff = np.array([apk(truth[u], preds[u]) - apk(truth[u], base[u]) for u in users])
    return {"users_up": int((diff > 1e-12).sum()), "users_down": int((diff < -1e-12).sum()),
            "users_same": int((np.abs(diff) <= 1e-12).sum())}


class Runner:
    def __init__(self, av, items_df, split, users, alpha):
        self.items_df, self.split, self.users, self.alpha = items_df, split, users, alpha
        self.train = P.train_views(av, split)
        self.pool = P.candidate_pool(items_df, split)
        self.popular = P.popularity_ranking(self.train, self.pool)
        self.end_week = P.SPLITS[split]["train_weeks"][1]
        self.graphs = {}

    def predict(self, beta: float = 0.0, half_life: float = math.inf):
        if half_life not in self.graphs:
            self.graphs[half_life] = ppr.fit_graph(self.train, half_life, self.end_week)
        graph = self.graphs[half_life]
        weight = airing_weight(self.items_df, graph["items"], self.split, beta)
        ranked, conv = ppr.rank_users(graph, self.pool, self.users, self.alpha, item_weight=weight)
        preds, fill = P.fill_to_k(ranked, self.users, self.popular, self.pool)
        return preds, fill, conv


def evaluate_variant(run: Runner, truth, cohorts, base, beta, half_life) -> tuple[dict, dict]:
    t0 = time.time()
    preds, fill, conv = run.predict(beta, half_life)
    row = {"beta": beta, "half_life": h_label(half_life), **P.evaluate(preds, truth, cohorts),
           "history_share_A": P.history_share(preds, run.train, cohorts["A"]),
           "vs_baseline_A": ap_change(preds, base, truth, cohorts["A"]),
           "identical_to_baseline_users": sum(preds[u] == base[u] for u in run.users),
           "converged": conv["all_converged"], "runtime_s": round(time.time() - t0, 2)}
    return row, preds


def neighbours(values: list, best) -> list:
    i = values.index(best)
    return values[max(i - 1, 0): i + 2]


def md5(path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def main():
    results = json.loads(P.RESULTS_FILE.read_text(encoding=config.ENCODING))
    alpha = results["methods"]["ppr"]["best_alpha"]
    baseline_map = results["methods"]["ppr"][SELECT_ON]
    av, items_df = P.load_airing_views(), P.load_items()
    truth = P.build_truth(av, "val")
    cohorts = P.build_cohorts(truth, P.submission_users())
    users = sorted(set(cohorts["A"]) | set(cohorts["C"]))
    base = load_preds(config.INTERIM_DIR / "preds_val_ppr.parquet")
    run = Runner(av, items_df, "val", users, alpha)

    exp1, exp2, all_preds = [], [], {}
    for beta in BETAS:
        row, preds = evaluate_variant(run, truth, cohorts, base, beta, math.inf)
        exp1.append(row)
        all_preds[(beta, math.inf)] = preds
        log.info("exp1 beta %-4g %s", beta, {k: row[k] for k in (SELECT_ON, "map@5_C", "vs_baseline_A")})
    for h in HALF_LIVES:
        row, preds = evaluate_variant(run, truth, cohorts, base, 0.0, h)  # H = inf computed again, not reused
        exp2.append(row)
        all_preds[(0.0, h)] = preds
        log.info("exp2 H %-4s %s", h_label(h), {k: row[k] for k in (SELECT_ON, "map@5_C", "vs_baseline_A")})

    checks = {"beta=0 equals current PPR (users)": exp1[0]["identical_to_baseline_users"],
              "H=inf equals current PPR (users)": exp2[0]["identical_to_baseline_users"],
              "users": len(users), "baseline_map@5_A_recomputed": exp1[0][SELECT_ON],
              "baseline_map@5_A_stored": baseline_map}
    assert checks["beta=0 equals current PPR (users)"] == len(users) and exp1[0][SELECT_ON] == baseline_map
    assert checks["H=inf equals current PPR (users)"] == len(users) and exp2[0][SELECT_ON] == baseline_map

    best1 = max(exp1, key=lambda r: r[SELECT_ON])  # first of equal = the baseline
    best2 = max(exp2, key=lambda r: r[SELECT_ON])
    gain1, gain2 = best1[SELECT_ON] > baseline_map, best2[SELECT_ON] > baseline_map
    exp3 = []
    if gain1 and gain2:
        for beta in neighbours(BETAS, best1["beta"]):
            for h in neighbours(HALF_LIVES, float(best2["half_life"])):
                row, preds = evaluate_variant(run, truth, cohorts, base, beta, h)
                exp3.append(row)
                all_preds[(beta, h)] = preds
                log.info("exp3 beta %-4g H %-4s %s", beta, h_label(h), {k: row[k] for k in (SELECT_ON, "map@5_C")})

    candidates = [exp1[0]] + exp1[1:] + exp2[1:] + exp3
    best = max(candidates, key=lambda r: r[SELECT_ON])
    improved = best[SELECT_ON] > baseline_map
    improve = {
        "baseline": {"method": "ppr", "alpha": alpha, SELECT_ON: baseline_map},
        "rule": "one variable per experiment; selection on MAP@5 A; replace baseline only if strictly better",
        "exp1_airing_prior": exp1, "exp2_recency": exp2,
        "exp3_combination": exp3 if exp3 else f"skipped: gain exp1={gain1}, exp2={gain2}",
        "checks": checks,
        "best": {k: best[k] for k in ("beta", "half_life", SELECT_ON, "map@5_C", "vs_baseline_A")},
        "improved": improved,
    }
    log.info("best %s; improved %s", improve["best"], improved)

    if improved:
        t0 = time.time()
        beta, h = best["beta"], float(best["half_life"])
        sub_users = P.submission_users()
        final_run = Runner(av, items_df, "test", sub_users, alpha)
        preds, fill, conv = final_run.predict(beta, h)
        path = config.SUBMISSIONS_DIR / "submission_v2.csv"
        write_submission(preds, path)
        errors = validate_submission(path, final_run.pool)
        improve["final_v2"] = {"alpha": alpha, "beta": beta, "half_life": h_label(h),
                               "train_weeks": list(P.SPLITS["test"]["train_weeks"]),
                               "n_airings_column": P.SPLITS["test"]["pool_col"], "fill": fill,
                               "converged": conv["all_converged"], "submission": path.name, "md5": md5(path),
                               "validation_errors": errors[:10], "runtime_s": round(time.time() - t0, 2)}
        log.info("v2: %s", improve["final_v2"])
        if errors:
            raise SystemExit(1)
    P.update_results("improve", "ppr", improve)


if __name__ == "__main__":
    main()
