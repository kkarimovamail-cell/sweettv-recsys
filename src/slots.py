"""Channel x time-slot habits on top of v5, validation only (compared with v5).

  slot         (channel_id, day of week x hour of the airing start) -> key = channel_id * 168 + dow * 24 + hour
  profile      p_u(slot) = sum_weeks w_week * max signal of u in that slot and week / sum_weeks w_week over
               every ISO week of the train window (also weeks without viewing); signal and weights as in v5:
               screen_time of pairs with screen_time >= T_MIN, w_week = 0.5 ** ((end_week - week) / H).
               p_u is in [0, 1]: a recency-weighted share of weeks in which u watched that channel at that hour.
  slot_ui      sum over labelled airings of programme i in the target window (schedule only: which programme
               airs on which channel when) of p_u(slot of the airing) = expected number of airings watched
  ranking      candidates = v5 top-50 followed by slot-only programmes of the slot top-50;
               score = slot / max_u(slot) + lambda * v5 / max_u(v5); lambda = inf -> v5 only;
               zero scores dropped, ties keep candidate order, popularity top-up
  slot_only    pure slot_ui ranking (ties: smaller id), reported separately

Run from the project root:  python -m src.slots
"""
import hashlib
import json
import math
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp

from src import config
from src import protocol as P
from src.cbf import novelty
from src.expected import candidates, held_out_and_bootstrap
from src.improve import ap_change
from src.join import canonical_ids, dedup_schedule
from src.signal_exp import make_runner
from src.submission import validate_submission, write_submission

LAMBDAS = [0.0, 0.1, 0.5, 1.0, 2.0, 4.0, math.inf]
DEPTH = 50
SLOTS_PER_WEEK = 168
SELECT_ON = f"map@{P.K}_A"

log = config.get_logger("15_slots")


def slot_key(channel: pd.Series, start: pd.Series) -> np.ndarray:
    return (channel.to_numpy(dtype=np.int64) * SLOTS_PER_WEEK
            + start.dt.dayofweek.to_numpy() * 24 + start.dt.hour.to_numpy())


def week_weight(week, end_week: int, half_life: float):
    return 0.5 ** ((end_week - np.asarray(week, dtype=float)) / half_life)


def slot_profile(train: pd.DataFrame, half_life: float, train_weeks: tuple[int, int]) -> pd.Series:
    """Series indexed by (user_id, key) -> recency-weighted share of train weeks with viewing in that slot."""
    w = train[train["watched"]]
    signal = w["signal"] if "signal" in w else pd.Series(1.0, index=w.index)
    per_week = (w.assign(key=slot_key(w["channel_id"], w["airing_start"]), sig=signal)
                .groupby(["user_id", "key", "week"])["sig"].max().reset_index())
    per_week["num"] = per_week["sig"] * week_weight(per_week["week"], train_weeks[1], half_life)
    denom = week_weight(np.arange(train_weeks[0], train_weeks[1] + 1), train_weeks[1], half_life).sum()
    return per_week.groupby(["user_id", "key"])["num"].sum() / denom


def target_airings(split: str) -> pd.DataFrame:
    """Labelled airings of the target window's ISO weeks, schedule only; counts per (programme, slot key)."""
    scheds = [canonical_ids(pd.read_parquet(config.INTERIM_DIR / f"schedule_{p}.parquet",
                                            columns=["channel_id", "start_time", "tv_show_id"]))
              for p in config.SCHEDULE_FILES]
    sched, _ = dedup_schedule(scheds)
    sched = sched[sched["tv_show_id"] != "0"]
    sched = sched[sched["start_time"].dt.isocalendar()["week"].between(*P.SPLITS[split]["target_weeks"])]
    return (sched.assign(key=slot_key(sched["channel_id"], sched["start_time"]))
            .groupby(["tv_show_id", "key"]).size().rename("n").reset_index())


def slot_matrix(profile: pd.Series, airings: pd.DataFrame, users: list[str]):
    """S (users with a profile x programmes) = slot_ui; columns in ascending numeric id order."""
    prof = profile[profile.index.get_level_values("user_id").isin(set(users))].reset_index()
    keys = pd.Index(pd.unique(np.concatenate([prof["key"].to_numpy(), airings["key"].to_numpy()])))
    items = np.array(sorted(airings["tv_show_id"].unique(), key=int))
    u_index = pd.Index(sorted(prof["user_id"].unique()))
    U = sp.csr_matrix((prof["num"].to_numpy(), (u_index.get_indexer(prof["user_id"]), keys.get_indexer(prof["key"]))),
                      shape=(len(u_index), len(keys)))
    A = sp.csr_matrix((airings["n"].to_numpy(dtype=float),
                       (keys.get_indexer(airings["key"]), pd.Index(items).get_indexer(airings["tv_show_id"]))),
                      shape=(len(keys), len(items)))
    return (U @ A).tocsr(), u_index, items


def slot_scores(profile: pd.Series, airings: pd.DataFrame, users: list[str], depth: int = DEPTH):
    """{user: ids}, {user: slot_ui} for the top-`depth` programmes with slot_ui > 0 (ties: smaller id)."""
    S, u_index, items = slot_matrix(profile, airings, users)
    ranked, scores = {}, {}
    for r, u in enumerate(u_index):
        cols, vals = S.indices[S.indptr[r]:S.indptr[r + 1]], S.data[S.indptr[r]:S.indptr[r + 1]]
        keep = vals > 0
        cols, vals = cols[keep], vals[keep]
        order = np.lexsort((cols, -vals))[:depth]
        ranked[u], scores[u] = items[cols[order]].tolist(), vals[order]
    return ranked, scores


def fuse(v5_ranked, v5_scores, slot_ranked, slot_scores_, lam: float, depth: int = P.K, with_scores: bool = False):
    out, out_scores = {}, {}
    for u in set(v5_ranked) | set(slot_ranked):
        a_ids, b_ids = v5_ranked.get(u, []), slot_ranked.get(u, [])
        a = dict(zip(a_ids, v5_scores[u] / v5_scores[u].max())) if a_ids else {}
        b = dict(zip(b_ids, slot_scores_[u] / slot_scores_[u].max())) if b_ids else {}
        cand = a_ids + [i for i in b_ids if i not in a]
        if math.isinf(lam):
            score = np.array([a.get(i, 0.0) for i in cand])
        else:
            score = np.array([b.get(i, 0.0) + lam * a.get(i, 0.0) for i in cand])
        order = [k for k in np.argsort(-score, kind="stable") if score[k] > 0][:depth]
        out[u] = [cand[k] for k in order]
        out_scores[u] = score[order]
    return (out, out_scores) if with_scores else out


def load_views() -> pd.DataFrame:
    return pd.read_parquet(config.INTERIM_DIR / "airing_views.parquet",
                           columns=P.AV_COLUMNS + ["screen_time", "channel_id", "airing_start"])


def build(av, items_df, split, users, alpha, v5):
    run = make_runner(av, items_df, split, users, alpha, v5["mode"], v5["t"])
    beta, half_life = v5["beta"], float(v5["half_life"])
    base, _, _ = run.predict(beta, half_life)
    _, v5_ranked, v5_sc = candidates(run, beta, half_life)
    profile = slot_profile(run.train, half_life, P.SPLITS[split]["train_weeks"])
    airings = target_airings(split)
    assert set(airings["tv_show_id"]) == run.pool, "target airings and pool differ"
    s_ranked, s_sc = slot_scores(profile, airings, users)
    return run, base, v5_ranked, v5_sc, s_ranked, s_sc, profile, airings


def main():
    results = json.loads(P.RESULTS_FILE.read_text(encoding=config.ENCODING))
    alpha = results["methods"]["ppr"]["best_alpha"]
    v5 = results["improve_signal"]["v2_signal"]["final_v5"]
    v5_map = results["improve_signal"]["v2_signal"]["table"][f"{v5['mode']}_{v5['t']:g}"][SELECT_ON]

    av, items_df = load_views(), P.load_items()
    truth = P.build_truth(av, "val")
    cohorts = P.build_cohorts(truth, P.submission_users())
    users = sorted(set(cohorts["A"]) | set(cohorts["C"]))
    protocol_train = P.train_views(av, "val")

    t0 = time.time()
    run, base, v5_ranked, v5_sc, s_ranked, s_sc, profile, airings = build(av, items_df, "val", users, alpha, v5)
    prep_s = time.time() - t0
    log.info("profile: %d (user, slot) cells for %d users; target airings %d in %d (programme, slot) cells",
             len(profile), profile.index.get_level_values(0).nunique(), int(airings["n"].sum()), len(airings))

    slot_preds, slot_fill = P.fill_to_k(s_ranked, users, run.popular, run.pool)
    slot_only = {**P.evaluate(slot_preds, truth, cohorts), **novelty(slot_preds, protocol_train, truth, cohorts["A"]),
                 "fill": slot_fill}
    log.info("slot_only %s", {k: slot_only[k] for k in (SELECT_ON, "map@5_C", "new_share_A", "new_slots_A", "new_slots_in_truth_A")})

    table = {}
    for lam in LAMBDAS:
        preds, fill = P.fill_to_k(fuse(v5_ranked, v5_sc, s_ranked, s_sc, lam), users, run.popular, run.pool)
        row = {"lambda": "inf" if math.isinf(lam) else lam, **P.evaluate(preds, truth, cohorts),
               **novelty(preds, protocol_train, truth, cohorts["A"]),
               "vs_v5_A": ap_change(preds, base, truth, cohorts["A"]),
               **held_out_and_bootstrap(preds, base, truth, cohorts),
               "identical_to_v5_users": sum(preds[u] == base[u] for u in users), "fill": fill}
        table[f"lambda={row['lambda']}"] = row
        log.info("lambda %-4s A %.6f C %.6f up/down %s dA %s C-A %.6f new %.4f same-as-v5 %d", row["lambda"],
                 row[SELECT_ON], row["map@5_C"], row["vs_v5_A"], row["delta_A_95ci"], row["map@5_C_minus_A"],
                 row["new_share_A"], row["identical_to_v5_users"])
    check = table["lambda=inf"]
    assert check["identical_to_v5_users"] == len(users) and check[SELECT_ON] == v5_map, "lambda=inf != v5"

    best_name = max(table, key=lambda n: (table[n][SELECT_ON], n == "lambda=inf"))  # tie -> keep v5
    best = table[best_name]
    accepted = best_name != "lambda=inf" and best[SELECT_ON] > v5_map and best["delta_A_95ci"][0] > 0
    prev_v6 = results.get("improve_slots", {}).get("v5_plus_slots", {}).get("final_v6")
    rebuild = accepted and (prev_v6 is None or prev_v6["lambda"] != best["lambda"])
    out = {"baseline_v5": {"alpha": alpha, **{k: v5[k] for k in ("beta", "half_life", "mode", "t")}, SELECT_ON: v5_map},
           "definition": __doc__.split("Run from")[0].strip(),
           "slot_only": slot_only, "table": table,
           "check_lambda_inf_equals_v5": {"users_identical": check["identical_to_v5_users"], "users": len(users)},
           "best": best_name, "rule": "accept only if MAP@5 A > v5 and the 95% CI of the A gain over v5 excludes 0",
           "accepted": accepted, "v6_rebuilt": rebuild,
           "v6_rule": "rebuild only if the best lambda changed and the 95% CI of its gain over v5 excludes 0",
           "prep_s": round(prep_s, 2)}
    if accepted and not rebuild:
        out["final_v6"] = prev_v6  # same lambda as before: file left untouched
    log.info("best %s accepted %s", best_name, accepted)

    if rebuild:
        t1 = time.time()
        lam = best["lambda"]
        sub_users = P.submission_users()
        run_f, _, r_f, s_f, sr_f, ss_f, _, _ = build(av, items_df, "test", sub_users, alpha, v5)
        preds_f, fill_f = P.fill_to_k(fuse(r_f, s_f, sr_f, ss_f, lam), sub_users, run_f.popular, run_f.pool)
        path = config.SUBMISSIONS_DIR / "submission_v6.csv"
        write_submission(preds_f, path)
        errors = validate_submission(path, run_f.pool)
        out["final_v6"] = {"alpha": alpha, **{k: v5[k] for k in ("beta", "half_life", "mode", "t")}, "lambda": lam,
                           "train_weeks": list(P.SPLITS["test"]["train_weeks"]), "fill": fill_f,
                           "submission": path.name, "md5": hashlib.md5(path.read_bytes()).hexdigest(),
                           "validation_errors": errors[:10], "runtime_s": round(time.time() - t1, 2)}
        log.info("v6: %s", out["final_v6"])
        if errors:
            raise SystemExit(1)
    P.update_results("improve_slots", "v5_plus_slots", out)


if __name__ == "__main__":
    main()
