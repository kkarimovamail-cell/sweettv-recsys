"""Expected-frequency re-ranking of PPR v2 candidates, validation only (compared with v2).

  candidates   top-DEPTH programmes of PPR v2 (alpha, beta, H from results_validation.json["improve"])
  rate_ui      watched_ui / n_airings_train_i: watched_ui = sum of recency weights 0.5 ** (delta / H) of the
               user's watched airings (same H as v2); n_airings_train_i = plain count of labelled airings
               of the programme in the train window (items.parquet)
  expected_ui  rate_ui * n_airings_target_i (labelled airings in the target window, schedule only)
  score        expected_ui + lambda * ppr_ui / max_u(ppr), sorted descending; remaining ties keep PPR order
  check        "ppr_only" (expected switched off) must reproduce v2 exactly

Run from the project root:  python -m src.expected
"""
import hashlib
import json
import time

import numpy as np

from src import config, ppr
from src import protocol as P
from src.improve import Runner, airing_weight, ap_change
from src.metrics import apk
from src.submission import validate_submission, write_submission

LAMBDAS = [0.0, 0.01, 0.1, 1.0]
DEPTH = 50
TRAIN_AIRING_COLS = {"val": ["n_airings_w11_20"], "test": ["n_airings_w11_20", "n_airings_w21_30"]}
SELECT_ON = f"map@{P.K}_A"
BOOTSTRAP = 2000

log = config.get_logger("12_expected")


def candidates(run: Runner, beta: float, half_life: float):
    """PPR v2 top-DEPTH ids and scores per user (also fills run.graphs[half_life])."""
    run.predict(beta, half_life)
    graph = run.graphs[half_life]
    weight = airing_weight(run.items_df, graph["items"], run.split, beta)
    ranked, _, scores = ppr.rank_users(graph, run.pool, run.users, run.alpha, depth=DEPTH,
                                       item_weight=weight, with_scores=True)
    return graph, ranked, scores


def expected_table(graph: dict, items_df, split: str, ranked: dict) -> dict:
    """{user: expected_ui for each candidate}, aligned with ranked[user]."""
    by_id = items_df.set_index("tv_show_id")
    n_train = by_id[TRAIN_AIRING_COLS[split]].sum(axis=1)
    n_target = by_id[P.SPLITS[split]["pool_col"]]
    item_pos = {i: k for k, i in enumerate(graph["items"])}
    H = graph["H"].tocsr()
    out = {}
    for u, ids in ranked.items():
        row = H[graph["users"].get_loc(u)]
        watched = dict(zip(row.indices, row.data))
        w = np.array([watched.get(item_pos.get(i, -1), 0.0) for i in ids])
        nt = n_train.reindex(ids).fillna(0).to_numpy(dtype=float)
        assert ((w == 0) | (nt > 0)).all(), f"watched programme without train airings for {u}"
        rate = np.divide(w, nt, out=np.zeros_like(w), where=nt > 0)
        out[u] = rate * n_target.reindex(ids).to_numpy(dtype=float)
    return out


def rerank(ranked: dict, ppr_scores: dict, expected: dict | None, lam: float) -> dict:
    out = {}
    for u, ids in ranked.items():
        norm = ppr_scores[u] / ppr_scores[u].max()
        score = norm if expected is None else expected[u] + lam * norm
        order = np.argsort(-score, kind="stable")  # stable -> ties keep PPR order
        out[u] = [ids[k] for k in order[:P.K]]
    return out


def held_out_and_bootstrap(preds, base, truth, cohorts) -> dict:
    a_set = set(cohorts["A"])
    held = [u for u in cohorts["C"] if u not in a_set]
    d = np.array([apk(truth[u], preds[u]) - apk(truth[u], base[u]) for u in cohorts["A"]])
    rng = np.random.default_rng(config.SEED)
    boot = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(BOOTSTRAP)])
    return {"delta_A_mean": round(float(d.mean()), 6),
            "delta_A_95ci": [round(float(x), 6) for x in np.percentile(boot, [2.5, 97.5])],
            "map@5_C_minus_A": round(float(np.mean([apk(truth[u], preds[u]) for u in held])), 6),
            "C_minus_A_users": len(held)}


def main():
    results = json.loads(P.RESULTS_FILE.read_text(encoding=config.ENCODING))
    alpha = results["methods"]["ppr"]["best_alpha"]
    v2 = results["improve"]["ppr"]["best"]
    beta, half_life, v2_map = v2["beta"], float(v2["half_life"]), v2[SELECT_ON]

    av, items_df = P.load_airing_views(), P.load_items()
    truth = P.build_truth(av, "val")
    cohorts = P.build_cohorts(truth, P.submission_users())
    users = sorted(set(cohorts["A"]) | set(cohorts["C"]))
    run = Runner(av, items_df, "val", users, alpha)

    t0 = time.time()
    base, _, _ = run.predict(beta, half_life)  # v2 on validation
    graph, ranked, scores = candidates(run, beta, half_life)
    expected = expected_table(graph, items_df, "val", ranked)
    prep_s = time.time() - t0

    variants = [("ppr_only (expected off)", None, None)] + [(f"lambda={lam:g}", lam, True) for lam in LAMBDAS]
    table = {}
    base_metrics = {**P.evaluate(base, truth, cohorts), **held_out_and_bootstrap(base, base, truth, cohorts)}
    for name, lam, use_expected in variants:
        t1 = time.time()
        fused = rerank(ranked, scores, expected if use_expected else None, lam or 0.0)
        preds, fill = P.fill_to_k(fused, users, run.popular, run.pool)
        row = {**P.evaluate(preds, truth, cohorts),
               "history_share_A": P.history_share(preds, run.train, cohorts["A"]),
               "vs_v2_A": ap_change(preds, base, truth, cohorts["A"]),
               **held_out_and_bootstrap(preds, base, truth, cohorts),
               "identical_to_v2_users": sum(preds[u] == base[u] for u in users),
               "fill": fill, "runtime_s": round(prep_s + time.time() - t1, 2)}
        table[name] = row
        log.info("%-24s A %.6f C %.6f up/down %s  dA %s  C-A %.6f  same-as-v2 %d",
                 name, row[SELECT_ON], row["map@5_C"], row["vs_v2_A"], row["delta_A_95ci"],
                 row["map@5_C_minus_A"], row["identical_to_v2_users"])

    check = table["ppr_only (expected off)"]
    assert check["identical_to_v2_users"] == len(users) and check[SELECT_ON] == v2_map, "ppr_only != v2"
    best_name = max((n for n in table if n.startswith("lambda")), key=lambda n: table[n][SELECT_ON])
    best = table[best_name]
    improved = best[SELECT_ON] > v2_map
    out = {
        "baseline_v2": {"alpha": alpha, "beta": beta, "half_life": v2["half_life"], SELECT_ON: v2_map,
                        "map@5_C": base_metrics["map@5_C"], "map@5_C_minus_A": base_metrics["map@5_C_minus_A"]},
        "definition": __doc__.split("Run from")[0].strip(),
        "hyperparameters": {"lambdas": LAMBDAS, "candidate_depth": DEPTH,
                            "ppr_normalisation": "score / max score among the user's candidates",
                            "rate_denominator": "plain count of train-window airings (not recency-weighted)"},
        "table": table,
        "check_ppr_only_equals_v2": {"users_identical": check["identical_to_v2_users"], "users": len(users)},
        "best": best_name, "improved": improved,
    }

    if improved:
        t2 = time.time()
        lam = float(best_name.split("=")[1])
        sub_users = P.submission_users()
        run_f = Runner(av, items_df, "test", sub_users, alpha)
        g_f, ranked_f, scores_f = candidates(run_f, beta, half_life)
        exp_f = expected_table(g_f, items_df, "test", ranked_f)
        preds_f, fill_f = P.fill_to_k(rerank(ranked_f, scores_f, exp_f, lam), sub_users, run_f.popular, run_f.pool)
        path = config.SUBMISSIONS_DIR / "submission_v3.csv"
        write_submission(preds_f, path)
        errors = validate_submission(path, run_f.pool)
        out["final_v3"] = {"alpha": alpha, "beta": beta, "half_life": v2["half_life"], "lambda": lam,
                           "train_weeks": list(P.SPLITS["test"]["train_weeks"]),
                           "train_airing_cols": TRAIN_AIRING_COLS["test"], "target_col": P.SPLITS["test"]["pool_col"],
                           "fill": fill_f, "submission": path.name,
                           "md5": hashlib.md5(path.read_bytes()).hexdigest(),
                           "validation_errors": errors[:10], "runtime_s": round(time.time() - t2, 2)}
        log.info("v3: %s", out["final_v3"])
        if errors:
            raise SystemExit(1)
    P.update_results("improve_expected", "ppr_v3", out)
    log.info("best %s improved %s", best_name, improved)


if __name__ == "__main__":
    main()
