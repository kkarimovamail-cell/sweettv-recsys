"""Baselines on the shared protocol.

  popularity:       same list for everyone, pool programmes ranked by distinct users with a watched airing in train
  personal_history: the user's own programmes ranked by watched airings in train (protocol tie rule),
                    topped up from popularity; reference only, not one of the two required methods

Run from the project root:  python -m src.baselines
"""
import time

from src import config
from src import protocol as P

log = config.get_logger("05_baselines")


def popularity_preds(train, pool, users):
    return P.fill_to_k({}, users, P.popularity_ranking(train, pool), pool)


def personal_history_preds(train, pool, users):
    counts = P.watched_counts(train)
    counts = counts[counts["tv_show_id"].isin(pool) & counts["user_id"].isin(set(users))]
    ranked = P.to_lists(P.rank_per_user(counts, P.K))
    return P.fill_to_k(ranked, users, P.popularity_ranking(train, pool), pool)


METHODS = {"popularity": popularity_preds, "personal_history": personal_history_preds}


def main():
    av, items = P.load_airing_views(), P.load_items()
    truth = P.build_truth(av, "val")
    cohorts = P.build_cohorts(truth, P.submission_users())
    users = sorted(set(cohorts["A"]) | set(cohorts["C"]))
    train, pool = P.train_views(av, "val"), P.candidate_pool(items, "val")

    for name, fn in METHODS.items():
        t0 = time.time()
        preds, fill_stats = fn(train, pool, users)
        runtime = round(time.time() - t0, 2)
        scores = P.evaluate(preds, truth, cohorts)
        scores["history_share_A"] = P.history_share(preds, train, cohorts["A"])
        P.save_preds(preds, config.INTERIM_DIR / f"preds_val_{name}.parquet")
        P.update_results("methods", name, {**scores, "fill": fill_stats, "runtime_s": runtime,
                                           "hyperparameters": {}, "seed": config.SEED})
        log.info("%-17s %s  fill=%s  %.2f s", name, scores, fill_stats, runtime)


if __name__ == "__main__":
    main()
