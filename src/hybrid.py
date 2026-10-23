"""Hybrid PPR + MF by reciprocal rank fusion, validation only.

  score(i) = w / (RRF_K + rank_PPR(i)) + (1 - w) / (RRF_K + rank_MF(i)), ranks 1-based within each
  method's own top-DEPTH list; a programme absent from a list gets 0 from that method.
  Programmes with score 0 are dropped (so w = 1 is pure PPR and w = 0 pure MF), ties -> smaller id,
  then top-up from popularity to 5. Components use the configurations selected in src/ppr.py and src/mf.py.

Run from the project root:  python -m src.hybrid   (after src.ppr and src.mf)
"""
import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import json  # noqa: E402
import time  # noqa: E402
from collections import defaultdict  # noqa: E402

import pandas as pd  # noqa: E402

from src import config  # noqa: E402
from src import mf, ppr  # noqa: E402
from src import protocol as P  # noqa: E402

WEIGHTS = [0.0, 0.25, 0.5, 0.75, 1.0]
RRF_K = 60
DEPTH = 50
SELECT_ON = f"map@{P.K}_A"

log = config.get_logger("09_hybrid")


def rrf(ranked_ppr: dict, ranked_mf: dict, users: list[str], w: float, k: int = RRF_K) -> dict[str, list[str]]:
    fused = {}
    for u in users:
        score = defaultdict(float)
        for r, i in enumerate(ranked_ppr.get(u, []), start=1):
            score[i] += w / (k + r)
        for r, i in enumerate(ranked_mf.get(u, []), start=1):
            score[i] += (1 - w) / (k + r)
        fused[u] = [i for i, s in sorted(score.items(), key=lambda x: (-x[1], int(x[0]))) if s > 0]
    return fused


def load_preds(path) -> dict[str, list[str]]:
    df = pd.read_parquet(path).sort_values(["user_id", "rank"])
    return df.groupby("user_id")["tv_show_id"].agg(list).to_dict()


def main():
    results = json.loads(P.RESULTS_FILE.read_text(encoding=config.ENCODING))
    alpha = results["methods"]["ppr"]["best_alpha"]
    mf_cfg = {k: results["methods"]["mf"]["grid"][results["methods"]["mf"]["best"]][k]
              for k in ("factors", "regularization", "alpha")}

    av, items = P.load_airing_views(), P.load_items()
    truth = P.build_truth(av, "val")
    cohorts = P.build_cohorts(truth, P.submission_users())
    users = sorted(set(cohorts["A"]) | set(cohorts["C"]))
    train, pool = P.train_views(av, "val"), P.candidate_pool(items, "val")
    popular = P.popularity_ranking(train, pool)

    t0 = time.time()
    ranked_ppr, _ = ppr.rank_users(ppr.fit_graph(train), pool, users, alpha, depth=DEPTH)
    ppr_s = time.time() - t0
    t0 = time.time()
    M, users_index, item_ids = mf.user_item_matrix(train)
    model = mf.fit_als(M, **mf_cfg)
    ranked_mf = mf.rank_users(model.user_factors, model.item_factors, users_index, item_ids, pool, users, depth=DEPTH)
    mf_s = time.time() - t0

    ref = {0.0: load_preds(config.INTERIM_DIR / "preds_val_mf.parquet"),
           1.0: load_preds(config.INTERIM_DIR / "preds_val_ppr.parquet")}
    grid, checks = {}, {}
    for w in WEIGHTS:
        t0 = time.time()
        preds, fill = P.fill_to_k(rrf(ranked_ppr, ranked_mf, users, w), users, popular, pool)
        fuse_s = time.time() - t0
        scores = P.evaluate(preds, truth, cohorts)
        scores["history_share_A"] = P.history_share(preds, train, cohorts["A"])
        grid[str(w)] = {**scores, "w": w, "fill": fill, "fusion_s": round(fuse_s, 2),
                        "runtime_s": round(ppr_s + mf_s + fuse_s, 2)}
        P.save_preds(preds, config.INTERIM_DIR / f"preds_val_hybrid_w{w:g}.parquet")
        if w in ref:
            same = sum(preds[u] == ref[w][u] for u in users)
            checks[f"w={w:g} equals {'MF' if w == 0 else 'PPR'}"] = {"users_identical": same, "users": len(users)}
        log.info("w %.2f  %s  fill %s", w, scores, fill)

    best = max(WEIGHTS, key=lambda w: grid[str(w)][SELECT_ON])  # first of equal scores = smaller w
    log.info("checks %s; best w %.2f", checks, best)
    P.update_results("methods", "hybrid_rrf", {
        **{k: grid[str(best)][k] for k in (f"map@{P.K}_A", f"map@{P.K}_C", "history_share_A")},
        "best_w": best, "selected_on": SELECT_ON, "grid": grid, "checks": checks,
        "components": {"ppr_alpha": alpha, "mf": mf_cfg, "ppr_rank_s": round(ppr_s, 2), "mf_fit_rank_s": round(mf_s, 2)},
        "hyperparameters": {"weights": WEIGHTS, "rrf_k": RRF_K, "depth_per_method": DEPTH},
        "seed": config.SEED,
    })


if __name__ == "__main__":
    main()
