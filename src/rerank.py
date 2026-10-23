"""Learned re-ranking on top of v6 with LightGBM lambdarank (validation + optional final).

Candidates per user: v6 top-50, then slot_ui top-50, then the user's pool programmes from the train window
(ordered by soft frequency), at most MAX_CANDIDATES. Every window-length dependent feature is per week
(train window: 10 weeks on validation, 20 on test; target window: 10 and 12 weeks).
History = train pairs with screen_time >= T_MIN (v5 signal); "hard" = screen_time >= 0.8.
Label = 5, 4, 3, 2, 1 for truth places 1-5, else 0.
Evaluation: features weeks 11-20, labels weeks 21-30, two folds of cohort-C users (seed 42), out-of-fold.

Run from the project root:  python -m src.rerank
"""
import hashlib
import json
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
import sklearn

from src import config, ppr
from src import protocol as P
from src.cbf import item_matrix, user_profiles
from src.expected import TRAIN_AIRING_COLS, candidates, held_out_and_bootstrap
from src.improve import airing_weight, ap_change
from src.signal_exp import make_runner
from src.slots import fuse, load_views, slot_matrix, slot_profile, slot_scores, target_airings
from src.submission import validate_submission, write_submission

DEPTH = 50
V5_FEATURE_DEPTH = 200
MAX_CANDIDATES = 100
HARD = 0.8
PARAMS = dict(objective="lambdarank", num_leaves=31, n_estimators=300, learning_rate=0.05,
              random_state=config.SEED, n_jobs=1, deterministic=True, force_row_wise=True, verbose=-1)
FEATURES = ["v6_rank", "v6_score", "v5_ppr", "slot_ui_pw", "slot_norm", "soft_freq_pw", "hard_freq_pw",
            "weeks_since_last", "n_airings_target_pw", "n_airings_train_pw", "watched_share", "popularity_pw",
            "category_share", "channel_share", "cbf_cosine", "user_activity_pw"]
SELECT_ON = f"map@{P.K}_A"

log = config.get_logger("16_rerank")


def n_weeks(weeks: tuple[int, int]) -> int:
    return weeks[1] - weeks[0] + 1


def lists_to_frame(ranked: dict, scores: dict, name: str, rank_name: str | None = None) -> pd.DataFrame:
    rows = [(u, i, s / sc[0], r + 1) for u, ids in ranked.items() if len(ids)
            for sc in [scores[u]] for r, (i, s) in enumerate(zip(ids, sc))]
    df = pd.DataFrame(rows, columns=["user_id", "tv_show_id", name, "_rank"])
    return df.rename(columns={"_rank": rank_name}) if rank_name else df.drop(columns="_rank")


def build_window(av, items_df, split, users, alpha, cfg, lam, V, ids):
    t0 = time.time()
    weeks, target = P.SPLITS[split]["train_weeks"], P.SPLITS[split]["target_weeks"]
    W, Wt, end_week = n_weeks(weeks), n_weeks(target), weeks[1]
    beta, half_life = cfg["beta"], float(cfg["half_life"])

    run = make_runner(av, items_df, split, users, alpha, cfg["mode"], cfg["t"])
    _, v5_r50, v5_s50 = candidates(run, beta, half_life)
    graph = run.graphs[half_life]
    v5_r, _, v5_s = ppr.rank_users(graph, run.pool, users, alpha, depth=V5_FEATURE_DEPTH,
                                   item_weight=airing_weight(items_df, graph["items"], split, beta), with_scores=True)
    profile = slot_profile(run.train, half_life, weeks)
    airings = target_airings(split)
    s_r50, s_s50 = slot_scores(profile, airings, users)
    S, s_users, s_items = slot_matrix(profile, airings, users)
    v6_deep, v6_deep_s = fuse(v5_r50, v5_s50, s_r50, s_s50, lam, depth=10**6, with_scores=True)
    v6_preds, _ = P.fill_to_k(v6_deep, users, run.popular, run.pool)
    v6_check, _ = P.fill_to_k(fuse(v5_r50, v5_s50, s_r50, s_s50, lam), users, run.popular, run.pool)
    assert v6_preds == v6_check

    tr = run.train  # soft-signal train: watched = screen_time >= T_MIN, signal = screen_time
    soft = tr[tr["watched"]]
    personal = soft.groupby(["user_id", "tv_show_id"]).agg(
        soft_sum=("signal", "sum"), n_soft=("signal", "size"), last_week=("week", "max")).reset_index()
    hard = tr[tr["screen_time"] >= HARD].groupby(["user_id", "tv_show_id"]).size().rename("n_hard").reset_index()
    history = {u: g.sort_values(["soft_sum", "tv_show_id"], ascending=[False, True])["tv_show_id"].tolist()
               for u, g in personal[personal["tv_show_id"].isin(run.pool)].groupby("user_id")}

    rows = []
    for u in users:
        cand = list(dict.fromkeys(v6_deep.get(u, [])[:DEPTH] + s_r50.get(u, []) + history.get(u, [])))[:MAX_CANDIDATES]
        rows += [(u, i, k) for k, i in enumerate(cand)]
    df = pd.DataFrame(rows, columns=["user_id", "tv_show_id", "cand_pos"])

    # ranking-model scores
    df = df.merge(lists_to_frame(v6_deep, v6_deep_s, "v6_score", "v6_rank"), how="left")
    df = df.merge(lists_to_frame(v5_r, v5_s, "v5_ppr"), how="left")
    df[["v6_score", "v5_ppr"]] = df[["v6_score", "v5_ppr"]].fillna(0.0)  # outside the list -> 0; v6_rank stays NaN
    u_pos, i_pos = s_users.get_indexer(df["user_id"]), pd.Index(s_items).get_indexer(df["tv_show_id"])
    ok = (u_pos >= 0) & (i_pos >= 0)
    slot = np.zeros(len(df))
    slot[ok] = np.asarray(S[u_pos[ok], i_pos[ok]]).ravel()
    df["slot_ui_pw"] = slot / Wt
    user_max = pd.Series({u: s[0] for u, s in s_s50.items() if len(s)})
    df["slot_norm"] = slot / df["user_id"].map(user_max).to_numpy()

    # personal history
    df = df.merge(personal, how="left").merge(hard, how="left")
    df["soft_freq_pw"] = df["soft_sum"].fillna(0) / W
    df["hard_freq_pw"] = df["n_hard"].fillna(0) / W
    df["weeks_since_last"] = end_week - df["last_week"]

    # programme airings and popularity
    by_id = items_df.set_index("tv_show_id")
    n_train = by_id[TRAIN_AIRING_COLS[split]].sum(axis=1)
    df["n_airings_target_pw"] = df["tv_show_id"].map(by_id[P.SPLITS[split]["pool_col"]]).to_numpy() / Wt
    df["n_airings_train_pw"] = df["tv_show_id"].map(n_train).to_numpy() / W
    df["watched_share"] = (df["n_soft"].fillna(0) / df["tv_show_id"].map(n_train).replace(0, np.nan)).clip(upper=1)
    protocol_train = P.train_views(av, split)
    pop = protocol_train[protocol_train["watched"]].groupby("tv_show_id").size()
    df["popularity_pw"] = df["tv_show_id"].map(pop).fillna(0).to_numpy() / W

    # category and channel shares of the user's soft-weighted history
    total = soft.groupby("user_id")["signal"].sum()
    cat = (soft.assign(category=soft["tv_show_id"].map(by_id["tv_show_category"]))
           .groupby(["user_id", "category"])["signal"].sum().rename("cat_w").reset_index())
    df["category"] = df["tv_show_id"].map(by_id["tv_show_category"])
    df = df.merge(cat, how="left", on=["user_id", "category"])
    df["category_share"] = df["cat_w"].fillna(0).to_numpy() / df["user_id"].map(total).to_numpy()
    main_channel = (airings_channels(split).sort_values(["tv_show_id", "n", "channel_id"], ascending=[True, False, True])
                    .drop_duplicates("tv_show_id").set_index("tv_show_id")["channel_id"])
    chan = (soft.groupby(["user_id", "channel_id"])["signal"].sum().rename("chan_w").reset_index()
            .rename(columns={"channel_id": "channel"}))
    df["channel"] = df["tv_show_id"].map(main_channel).astype("int64")
    df = df.merge(chan.astype({"channel": "int64"}), how="left", on=["user_id", "channel"])
    df["channel_share"] = df["chan_w"].fillna(0).to_numpy() / df["user_id"].map(total).to_numpy()

    # content similarity (profiles as in method 3: protocol watched, recency H)
    prof, prof_users = user_profiles(V, ids, protocol_train, half_life, end_week)
    df["cbf_cosine"] = cbf_pairs(prof, prof_users, V, ids, df)
    df["user_activity_pw"] = df["user_id"].map(total).fillna(0).to_numpy() / W

    stats = {"split": split, "train_weeks": list(weeks), "target_weeks": list(target), "users": len(users),
             "candidates": len(df), "users_with_candidates": int(df["user_id"].nunique()),
             "mean_candidates": round(len(df) / max(df["user_id"].nunique(), 1), 1), "build_s": round(time.time() - t0, 2)}
    return {"df": df, "v6_preds": v6_preds, "run": run, "stats": stats}


def airings_channels(split: str) -> pd.DataFrame:
    """Labelled airings per (programme, channel) in the target window (schedule only)."""
    from src.join import canonical_ids, dedup_schedule
    scheds = [canonical_ids(pd.read_parquet(config.INTERIM_DIR / f"schedule_{p}.parquet",
                                            columns=["channel_id", "start_time", "tv_show_id"]))
              for p in config.SCHEDULE_FILES]
    s, _ = dedup_schedule(scheds)
    s = s[(s["tv_show_id"] != "0") & s["start_time"].dt.isocalendar()["week"].between(*P.SPLITS[split]["target_weeks"])]
    return s.groupby(["tv_show_id", "channel_id"]).size().rename("n").reset_index()


def cbf_pairs(prof, prof_users: pd.Index, V, ids: np.ndarray, df: pd.DataFrame) -> np.ndarray:
    out = np.zeros(len(df))
    u_pos = prof_users.get_indexer(df["user_id"])
    i_pos = pd.Index(ids).get_indexer(df["tv_show_id"])
    known = np.flatnonzero(u_pos >= 0)
    for b in range(0, len(known), 200_000):
        k = known[b:b + 200_000]
        out[k] = np.asarray(prof[u_pos[k]].multiply(V[i_pos[k]]).sum(axis=1)).ravel()
    return out


def add_labels(df: pd.DataFrame, truth: dict) -> pd.DataFrame:
    lab = pd.DataFrame([(u, i, P.K - r) for u, ids in truth.items() for r, i in enumerate(ids)],
                       columns=["user_id", "tv_show_id", "label"])
    return df.merge(lab, how="left").fillna({"label": 0}).astype({"label": "int32"})


def rank_by(df: pd.DataFrame, score: np.ndarray, users, run) -> dict:
    d = df.assign(_s=score).sort_values(["user_id", "_s", "cand_pos"], ascending=[True, False, True])
    ranked = d.groupby("user_id", sort=False)["tv_show_id"].agg(list).to_dict()
    return P.fill_to_k(ranked, users, run.popular, run.pool)[0]


def fit(df: pd.DataFrame) -> lgb.LGBMRanker:
    d = df.sort_values(["user_id", "cand_pos"])
    model = lgb.LGBMRanker(**PARAMS)
    model.fit(d[FEATURES], d["label"], group=d.groupby("user_id", sort=False).size().to_numpy())
    return model


def importance(models) -> dict:
    out = {}
    for kind in ("gain", "split"):
        vals = np.mean([m.booster_.feature_importance(importance_type=kind) for m in models], axis=0)
        out[kind] = {f: round(float(v), 2) for f, v in sorted(zip(FEATURES, vals), key=lambda x: -x[1])}
    return out


def main():
    t0 = time.time()
    results = json.loads(P.RESULTS_FILE.read_text(encoding=config.ENCODING))
    alpha = results["methods"]["ppr"]["best_alpha"]
    v5 = results["improve_signal"]["v2_signal"]["final_v5"]
    cfg = {k: v5[k] for k in ("beta", "half_life", "mode", "t")}
    slots_res = results["improve_slots"]["v5_plus_slots"]
    lam = float(slots_res["table"][slots_res["best"]]["lambda"])
    v6_map = slots_res["table"][slots_res["best"]][SELECT_ON]

    av, items_df = load_views(), P.load_items()
    V, ids, _ = item_matrix(items_df)
    truth = P.build_truth(av, "val")
    cohorts = P.build_cohorts(truth, P.submission_users())
    users = cohorts["C"]
    win = build_window(av, items_df, "val", users, alpha, cfg, lam, V, ids)
    df, v6_preds, run = add_labels(win["df"], truth), win["v6_preds"], win["run"]
    log.info("val window %s", win["stats"])

    in_cand = df[df["label"] > 0].groupby("user_id").size()
    recall = {"truth_items_in_candidates_C": round(float(in_cand.reindex(users).fillna(0).sum() / (P.K * len(users))), 4),
              "truth_items_in_v6_top5_C": round(float(np.mean([len(set(v6_preds[u]) & set(truth[u])) for u in users]) / P.K), 4)}
    log.info("candidate recall %s", recall)

    # check: ranking by the v6 score feature alone reproduces v6
    check_ranked = (df[df["v6_rank"].notna()].sort_values(["user_id", "v6_score", "cand_pos"], ascending=[True, False, True])
                    .groupby("user_id", sort=False)["tv_show_id"].agg(list).to_dict())
    v6_from_feature, _ = P.fill_to_k(check_ranked, users, run.popular, run.pool)
    same = sum(v6_from_feature[u] == v6_preds[u] for u in users)
    v6_eval = P.evaluate(v6_preds, truth, cohorts)
    assert same == len(users) and v6_eval[SELECT_ON] == v6_map, f"v6 feature ranking differs for {len(users) - same} users"
    log.info("check: v6_score ranking equals v6 for %d / %d users; v6 MAP@5 A %.6f", same, len(users), v6_eval[SELECT_ON])

    # two folds over cohort-C users, out-of-fold predictions
    perm = np.random.default_rng(config.SEED).permutation(np.array(sorted(users)))
    folds = [set(perm[: len(perm) // 2]), set(perm[len(perm) // 2:])]
    oof, models, fold_s = {}, [], []
    for k in range(2):
        t1 = time.time()
        train_df, test_df = df[df["user_id"].isin(folds[k])], df[df["user_id"].isin(folds[1 - k])]
        model = fit(train_df)
        models.append(model)
        fold_users = sorted(folds[1 - k])
        oof.update(rank_by(test_df, model.predict(test_df[FEATURES]), fold_users, run))
        fold_s.append(round(time.time() - t1, 2))
    row = {**P.evaluate(oof, truth, cohorts), "vs_v6_A": ap_change(oof, v6_preds, truth, cohorts["A"]),
           **held_out_and_bootstrap(oof, v6_preds, truth, cohorts),
           "history_share_A": P.history_share(oof, P.train_views(av, "val"), cohorts["A"])}
    imp = importance(models)
    log.info("OOF %s", row)
    log.info("importance (gain) %s", imp["gain"])

    # reproducibility of one fold model
    again = fit(df[df["user_id"].isin(folds[0])])
    probe = df[df["user_id"].isin(folds[1])][FEATURES]
    repro = bool(np.array_equal(models[0].predict(probe), again.predict(probe)))

    accepted = row[SELECT_ON] > v6_map and row["delta_A_95ci"][0] > 0
    out = {"baseline_v6": {"alpha": alpha, **cfg, "lambda": lam, SELECT_ON: v6_map, "map@5_C": v6_eval["map@5_C"]},
           "definition": __doc__.split("Run from")[0].strip(),
           "features": FEATURES, "params": PARAMS, "versions": {"lightgbm": lgb.__version__, "scikit-learn": sklearn.__version__},
           "window_val": win["stats"], "candidate_recall": recall,
           "check_v6_score_feature_equals_v6": {"users_identical": same, "users": len(users)},
           "folds": {"users": [len(f) for f in folds], "seed": config.SEED, "fit_predict_s": fold_s},
           "oof": row, "importance_mean_of_folds": imp, "reproducible_fold_model": repro,
           "rule": "accept only if OOF MAP@5 A > v6 and the 95% CI of the A gain over v6 excludes 0",
           "accepted": accepted}
    log.info("accepted %s (reproducible fold model: %s)", accepted, repro)

    if accepted:
        t2 = time.time()
        final_model = fit(df)
        out["importance_final_model"] = importance([final_model])
        sub_users = P.submission_users()
        test_win = build_window(av, items_df, "test", sub_users, alpha, cfg, lam, V, ids)
        tdf = test_win["df"]
        preds_f = rank_by(tdf, final_model.predict(tdf[FEATURES]), sub_users, test_win["run"])
        path = config.SUBMISSIONS_DIR / "submission_v7.csv"
        write_submission(preds_f, path)
        errors = validate_submission(path, test_win["run"].pool)
        out["final_v7"] = {"trained_on": f"{len(users)} cohort-C users, features weeks 11-20, labels weeks 21-30",
                           "window_test": test_win["stats"], "submission": path.name,
                           "md5": hashlib.md5(path.read_bytes()).hexdigest(), "validation_errors": errors[:10],
                           "runtime_s": round(time.time() - t2, 2)}
        log.info("v7: %s", out["final_v7"])
        if errors:
            raise SystemExit(1)
    out["runtime_s"] = round(time.time() - t0, 2)
    P.update_results("rerank", "lgbm_v6", out)


if __name__ == "__main__":
    main()
