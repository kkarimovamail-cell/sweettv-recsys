"""Last refinement of the v7 re-ranker, same protocol (out-of-fold over two folds of cohort-C users).

  exp 1  label: binary (1 = programme in the truth) vs graded 5..1 (v7)
  exp 2  averaging: 5 models with subsample 0.8, subsample_freq 1, colsample_bytree 0.8, random_state 42-46;
         per user each model ranks candidates (rank 1 = best, ties -> candidate order), the mean rank is
         sorted ascending (ties -> candidate order). Compared with one model.
A change is kept only if the 95% CI of its MAP@5 A gain excludes 0 (exp 1 against v7; exp 2 against the
best configuration after exp 1, which is v7 when exp 1 is not kept).

Run from the project root:  python -m src.rerank_v8   (after src.rerank)
"""
import hashlib
import json
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from src import config
from src import protocol as P
from src.cbf import item_matrix
from src.expected import held_out_and_bootstrap
from src.improve import ap_change
from src.rerank import FEATURES, PARAMS, add_labels, build_window, rank_by
from src.slots import load_views
from src.submission import validate_submission, write_submission

ENSEMBLE_SEEDS = [42, 43, 44, 45, 46]
BAGGING = dict(subsample=0.8, subsample_freq=1, colsample_bytree=0.8)
SELECT_ON = f"map@{P.K}_A"

log = config.get_logger("17_rerank_v8")


def label_column(df: pd.DataFrame, kind: str) -> pd.Series:
    return df["label"] if kind == "graded" else (df["label"] > 0).astype("int32")


def fit_model(df: pd.DataFrame, label: str, params: dict) -> lgb.LGBMRanker:
    d = df.sort_values(["user_id", "cand_pos"])
    model = lgb.LGBMRanker(**params)
    model.fit(d[FEATURES], label_column(d, label), group=d.groupby("user_id", sort=False).size().to_numpy())
    return model


def within_user_rank(df: pd.DataFrame, score: np.ndarray) -> np.ndarray:
    """1 = best per user; ties -> candidate order. Aligned with df rows."""
    d = df[["user_id", "cand_pos"]].assign(_s=score, _row=np.arange(len(df)))
    d = d.sort_values(["user_id", "_s", "cand_pos"], ascending=[True, False, True])
    rank = np.empty(len(df))
    rank[d["_row"].to_numpy()] = d.groupby("user_id", sort=False).cumcount().to_numpy() + 1
    return rank


def predict_score(models, df: pd.DataFrame) -> np.ndarray:
    """Higher = better: the model score for one model, minus the mean within-user rank for several."""
    if len(models) == 1:
        return models[0].predict(df[FEATURES])
    return -np.mean([within_user_rank(df, m.predict(df[FEATURES])) for m in models], axis=0)


def configs(label: str) -> dict:
    return {"single": [PARAMS],
            "ensemble": [{**PARAMS, **BAGGING, "random_state": s} for s in ENSEMBLE_SEEDS],
            "single_bagged_seed42": [{**PARAMS, **BAGGING, "random_state": 42}]}


def oof(df, folds, run, label: str, params_list: list[dict]) -> tuple[dict, float]:
    t0, preds = time.time(), {}
    for k in range(2):
        train_df, test_df = df[df["user_id"].isin(folds[k])], df[df["user_id"].isin(folds[1 - k])]
        models = [fit_model(train_df, label, p) for p in params_list]
        preds.update(rank_by(test_df, predict_score(models, test_df), sorted(folds[1 - k]), run))
    return preds, round(time.time() - t0, 2)


def score_row(preds, ref, truth, cohorts, ref_name: str) -> dict:
    boot = held_out_and_bootstrap(preds, ref, truth, cohorts)
    return {**P.evaluate(preds, truth, cohorts), f"vs_{ref_name}_A": ap_change(preds, ref, truth, cohorts["A"]),
            f"delta_vs_{ref_name}_mean": boot["delta_A_mean"], f"delta_vs_{ref_name}_95ci": boot["delta_A_95ci"],
            "map@5_C_minus_A": boot["map@5_C_minus_A"]}


def main():
    t_start = time.time()
    results = json.loads(P.RESULTS_FILE.read_text(encoding=config.ENCODING))
    alpha = results["methods"]["ppr"]["best_alpha"]
    v5 = results["improve_signal"]["v2_signal"]["final_v5"]
    cfg = {k: v5[k] for k in ("beta", "half_life", "mode", "t")}
    s = results["improve_slots"]["v5_plus_slots"]
    lam = float(s["table"][s["best"]]["lambda"])
    v7_map = results["rerank"]["lgbm_v6"]["oof"][SELECT_ON]

    av, items_df = load_views(), P.load_items()
    V, ids, _ = item_matrix(items_df)
    truth = P.build_truth(av, "val")
    cohorts = P.build_cohorts(truth, P.submission_users())
    users = cohorts["C"]
    win = build_window(av, items_df, "val", users, alpha, cfg, lam, V, ids)
    df, run = add_labels(win["df"], truth), win["run"]
    perm = np.random.default_rng(config.SEED).permutation(np.array(sorted(users)))
    folds = [set(perm[: len(perm) // 2]), set(perm[len(perm) // 2:])]

    # reference: v7 reproduced (graded labels, one deterministic model)
    v7, v7_s = oof(df, folds, run, "graded", configs("graded")["single"])
    v7_eval = P.evaluate(v7, truth, cohorts)
    assert v7_eval[SELECT_ON] == v7_map, f"v7 not reproduced: {v7_eval[SELECT_ON]} vs {v7_map}"
    table = {"v7 (graded, single)": {**v7_eval, "runtime_s": v7_s}}
    log.info("v7 reproduced: %s", v7_eval)

    # exp 1: binary label, one model
    binary, b_s = oof(df, folds, run, "binary", configs("binary")["single"])
    table["binary, single"] = {**score_row(binary, v7, truth, cohorts, "v7"), "runtime_s": b_s}
    keep_binary = table["binary, single"][SELECT_ON] > v7_map and table["binary, single"]["delta_vs_v7_95ci"][0] > 0
    label = "binary" if keep_binary else "graded"
    best_preds, best_name = (binary, "binary, single") if keep_binary else (v7, "v7 (graded, single)")
    log.info("exp1 binary %s -> keep %s", table["binary, single"], keep_binary)

    # exp 2: 5-model averaging (and one bagged model for context), with the label kept after exp 1
    for name in ("ensemble", "single_bagged_seed42"):
        preds, t = oof(df, folds, run, label, configs(label)[name])
        row = {**score_row(preds, v7, truth, cohorts, "v7"), "runtime_s": t}
        if best_name != "v7 (graded, single)":
            row.update({k: v for k, v in score_row(preds, best_preds, truth, cohorts, "best_after_exp1").items()
                        if "best_after_exp1" in k})
        table[f"{label}, {name}"] = row
        log.info("exp2 %s, %s: %s", label, name, row)
    ens = table[f"{label}, ensemble"]
    ci_key = "delta_vs_best_after_exp1_95ci" if best_name != "v7 (graded, single)" else "delta_vs_v7_95ci"
    keep_ensemble = ens[SELECT_ON] > table[best_name][SELECT_ON] and ens[ci_key][0] > 0
    final_name = f"{label}, ensemble" if keep_ensemble else best_name

    improved = final_name != "v7 (graded, single)"
    out = {"baseline_v7": {SELECT_ON: v7_map, "params": PARAMS},
           "definition": __doc__.split("Run from")[0].strip(), "bagging": BAGGING, "ensemble_seeds": ENSEMBLE_SEEDS,
           "param_names_checked": "subsample, subsample_freq, colsample_bytree, random_state in LGBMRanker.__init__ (lightgbm "
                                  f"{lgb.__version__})",
           "table": table, "keep_binary_label": keep_binary, "keep_ensemble": keep_ensemble,
           "final": final_name, "improved_over_v7": improved}
    log.info("keep binary %s, keep ensemble %s -> final %s", keep_binary, keep_ensemble, final_name)

    if improved:
        t1 = time.time()
        params_list = configs(label)["ensemble" if keep_ensemble else "single"]
        models = [fit_model(df, label, p) for p in params_list]
        sub_users = P.submission_users()
        test_win = build_window(av, items_df, "test", sub_users, alpha, cfg, lam, V, ids)
        tdf = test_win["df"]
        preds_f = rank_by(tdf, predict_score(models, tdf), sub_users, test_win["run"])
        path = config.SUBMISSIONS_DIR / "submission_v8.csv"
        write_submission(preds_f, path)
        errors = validate_submission(path, test_win["run"].pool)
        out["final_v8"] = {"label": label, "models": len(models), "trained_on": f"{len(users)} cohort-C users, "
                           "features weeks 11-20, labels weeks 21-30", "window_test": test_win["stats"],
                           "submission": path.name, "md5": hashlib.md5(path.read_bytes()).hexdigest(),
                           "validation_errors": errors[:10], "runtime_s": round(time.time() - t1, 2)}
        log.info("v8: %s", out["final_v8"])
        if errors:
            raise SystemExit(1)
    out["runtime_s"] = round(time.time() - t_start, 2)
    P.update_results("rerank_v8", "v7_refinement", out)


if __name__ == "__main__":
    main()
