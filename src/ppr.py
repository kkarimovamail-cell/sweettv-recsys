"""Personalized PageRank on the programme co-view graph, on the shared protocol.

  X        user x item, 1 if the user has a watched airing of the programme in the train window
  W        = X^T X, diagonal set to 0, each row keeps its TOP_NEIGHBOURS largest weights
           (ties: smaller programme id), P = row-normalised W; rows with no neighbours are dangling
  restart  v = the user's watched-airing counts per programme, normalised to sum 1
  iterate  p <- alpha * (P^T p + d * v) + (1 - alpha) * v,  d = mass on dangling nodes, p0 = v,
           until max_user ||p_new - p||_1 < TOL or MAX_ITER
  output   pool programmes with p > 0 by descending p (ties: smaller id), top-up from popularity to 5

Run from the project root:  python -m src.ppr
"""
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp

from src import config
from src import protocol as P
from src.submission import validate_submission, write_submission

ALPHAS = [0.1, 0.2, 0.3, 0.5, 0.7, 0.85]
TOP_NEIGHBOURS = 50
TOL = 1e-6
MAX_ITER = 200
BATCH_USERS = 1024
SELECT_ON = f"map@{P.K}_A"

log = config.get_logger("07_ppr")


def interaction_matrices(train: pd.DataFrame, half_life: float | None = None, end_week: int | None = None):
    """X for the graph and H for the restart (users x items); items sorted by numeric id.

    half_life None: X binary, H = watched-airing counts.
    half_life h:    each watched airing weighs 0.5 ** ((end_week - week) / h);
                    X = max weight per (user, programme) = the most recent watched airing (1 when h is infinite),
                    H = sum of weights.
    An optional "signal" column multiplies each watched airing's weight (e.g. screen_time as a soft weight).
    """
    w = train[train["watched"]]
    weight = (np.ones(len(w)) if half_life is None
              else 0.5 ** ((end_week - w["week"].to_numpy(dtype=float)) / half_life))
    if "signal" in w:
        weight = weight * w["signal"].to_numpy(dtype=float)
    agg = w.assign(weight=weight).groupby(["user_id", "tv_show_id"], as_index=False).agg(
        x=("weight", "max"), h=("weight", "sum"))
    items = sorted(agg["tv_show_id"].unique(), key=int)
    users = sorted(agg["user_id"].unique())
    r = pd.Index(users).get_indexer(agg["user_id"])
    c = pd.Index(items).get_indexer(agg["tv_show_id"])
    shape = (len(users), len(items))
    X = sp.csr_matrix((agg["x"].to_numpy(dtype=float), (r, c)), shape=shape)
    H = sp.csr_matrix((agg["h"].to_numpy(dtype=float), (r, c)), shape=shape)
    return X, H, users, items


def coview_transition(X: sp.csr_matrix, top_k: int = TOP_NEIGHBOURS):
    W = (X.T @ X).tocsr()
    W.setdiag(0)
    W.eliminate_zeros()
    rows, cols, vals = [], [], []
    for i in range(W.shape[0]):  # one pass over items, not over log rows
        idx, val = W.indices[W.indptr[i]:W.indptr[i + 1]], W.data[W.indptr[i]:W.indptr[i + 1]]
        keep = np.lexsort((idx, -val))[:top_k]  # largest weight first, ties -> lower index = smaller id
        rows.append(np.full(len(keep), i))
        cols.append(idx[keep])
        vals.append(val[keep])
    Wk = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=W.shape)
    row_sum = np.asarray(Wk.sum(axis=1)).ravel()
    dangling = row_sum == 0
    Pm = (sp.diags(np.divide(1.0, row_sum, out=np.zeros_like(row_sum), where=~dangling)) @ Wk).tocsr()
    p_rows = np.asarray(Pm.sum(axis=1)).ravel()
    stats = {"items": W.shape[0], "edges_before_top_k": int(W.nnz), "edges_after_top_k": int(Pm.nnz),
             "dangling_items": int(dangling.sum()),
             "max_abs_row_sum_minus_1": float(np.abs(p_rows[~dangling] - 1).max()),
             "dangling_row_sum_max": float(p_rows[dangling].max()) if dangling.any() else 0.0}
    return Pm, dangling, stats


def ppr(PT: sp.csr_matrix, dangling: np.ndarray, V: np.ndarray, alpha: float,
        tol: float = TOL, max_iter: int = MAX_ITER):
    """Power iteration for a batch: V is items x users with columns summing to 1."""
    p, one_step_mass_err = V.copy(), None
    for it in range(1, max_iter + 1):
        d = p[dangling].sum(axis=0)
        new = alpha * (PT @ p + V * d) + (1 - alpha) * V
        if one_step_mass_err is None:
            one_step_mass_err = float(np.abs(new.sum(axis=0) - p.sum(axis=0)).max())
        delta = float(np.abs(new - p).sum(axis=0).max())
        p = new
        if delta < tol:
            return p, it, True, one_step_mass_err
    return p, max_iter, False, one_step_mass_err


def fit_graph(train: pd.DataFrame, half_life: float | None = None, end_week: int | None = None) -> dict:
    X, H, users, items = interaction_matrices(train, half_life, end_week)
    Pm, dangling, stats = coview_transition(X)
    return {"PT": Pm.T.tocsr(), "dangling": dangling, "H": H, "users": pd.Index(users),
            "items": np.array(items), "stats": stats}


def rank_users(graph: dict, pool: set[str], users: list[str], alpha: float, depth: int = P.K,
               item_weight: np.ndarray | None = None, with_scores: bool = False):
    """Up to `depth` pool programmes with score > 0 per known user, by descending score (ties: smaller id).
    score = p, or p * item_weight (aligned with graph["items"]) when given.
    with_scores=True also returns {user: scores aligned with the ranked ids}."""
    pool_idx = np.flatnonzero(np.isin(graph["items"], list(pool)))
    pos = graph["users"].get_indexer(users)
    known = [(u, p) for u, p in zip(users, pos) if p >= 0]
    ranked, scores, iters, converged, mass_err, final_mass_err = {}, {}, [], [], [], []
    for b in range(0, len(known), BATCH_USERS):
        batch = known[b:b + BATCH_USERS]
        Hb = graph["H"][[p for _, p in batch]].toarray().T
        V = Hb / Hb.sum(axis=0, keepdims=True)
        p, it, conv, err = ppr(graph["PT"], graph["dangling"], V, alpha)
        iters.append(it)
        converged.append(conv)
        mass_err.append(err)
        final_mass_err.append(float(np.abs(p.sum(axis=0) - 1).max()))
        S = p[pool_idx] if item_weight is None else p[pool_idx] * item_weight[pool_idx][:, None]
        order = np.argsort(-S, axis=0, kind="stable")[:depth]  # stable -> ties keep ascending id order
        for j, (u, _) in enumerate(batch):
            top = order[:, j]
            top = top[S[top, j] > 0]
            ranked[u] = graph["items"][pool_idx[top]].tolist()
            if with_scores:
                scores[u] = S[top, j]
    conv = {"batches": len(iters), "iterations": iters, "all_converged": all(converged),
            "one_step_mass_change_max": max(mass_err), "final_mass_error_max": max(final_mass_err)}
    return (ranked, conv, scores) if with_scores else (ranked, conv)


def recommend(graph: dict, pool: set[str], users: list[str], alpha: float, popular: list[str]):
    ranked, conv = rank_users(graph, pool, users, alpha)
    preds, fill = P.fill_to_k(ranked, users, popular, pool)
    return preds, fill, conv


def main():
    av, items = P.load_airing_views(), P.load_items()
    truth = P.build_truth(av, "val")
    cohorts = P.build_cohorts(truth, P.submission_users())
    users = sorted(set(cohorts["A"]) | set(cohorts["C"]))
    train, pool = P.train_views(av, "val"), P.candidate_pool(items, "val")

    t0 = time.time()
    graph = fit_graph(train)
    graph_s = round(time.time() - t0, 2)
    log.info("val graph: %s (%.2f s)", graph["stats"], graph_s)
    popular = P.popularity_ranking(train, pool)

    grid, all_preds = {}, {}
    for alpha in ALPHAS:
        t1 = time.time()
        preds, fill, conv = recommend(graph, pool, users, alpha, popular)
        scores = P.evaluate(preds, truth, cohorts)
        scores["history_share_A"] = P.history_share(preds, train, cohorts["A"])
        grid[str(alpha)] = {**scores, "fill": fill, "convergence": conv, "runtime_s": round(time.time() - t1, 2)}
        all_preds[alpha] = preds
        P.save_preds(preds, config.INTERIM_DIR / f"preds_val_ppr_a{alpha:g}.parquet")
        log.info("alpha %.2f  %s  iters %s  converged %s  mass(1 step) %.2e  %.2f s", alpha, scores,
                 conv["iterations"], conv["all_converged"], conv["one_step_mass_change_max"], grid[str(alpha)]["runtime_s"])

    best = max(ALPHAS, key=lambda a: grid[str(a)][SELECT_ON])  # first of equal scores = smaller alpha
    P.save_preds(all_preds[best], config.INTERIM_DIR / "preds_val_ppr.parquet")
    P.update_results("methods", "ppr", {
        **{k: grid[str(best)][k] for k in (f"map@{P.K}_A", f"map@{P.K}_C", "history_share_A")},
        "best_alpha": best, "selected_on": SELECT_ON, "grid": grid,
        "hyperparameters": {"alphas": ALPHAS, "top_neighbours": TOP_NEIGHBOURS, "tol": TOL,
                            "max_iter": MAX_ITER, "batch_users": BATCH_USERS,
                            "restart": "watched-airing counts per programme, normalised"},
        "graph": {**graph["stats"], "build_s": graph_s}, "seed": config.SEED,
    })
    log.info("best alpha %.2f by %s", best, SELECT_ON)

    # final: train on weeks 11-30, predict for submission users, pool of weeks 31-42
    t2 = time.time()
    train_f, pool_f = P.train_views(av, "test"), P.candidate_pool(items, "test")
    graph_f = fit_graph(train_f)
    sub_users = P.submission_users()
    preds_f, fill_f, conv_f = recommend(graph_f, pool_f, sub_users, best, P.popularity_ranking(train_f, pool_f))
    path = config.SUBMISSIONS_DIR / "submission_ppr.csv"
    write_submission(preds_f, path)
    errors = validate_submission(path, pool_f)
    P.update_results("final", "ppr", {"alpha": best, "train_weeks": list(P.SPLITS["test"]["train_weeks"]),
                                      "graph": graph_f["stats"], "fill": fill_f, "convergence": conv_f,
                                      "submission": path.name, "validation_errors": errors[:10],
                                      "runtime_s": round(time.time() - t2, 2)})
    log.info("final graph %s; fill %s; converged %s; %s: %s", graph_f["stats"], fill_f,
             conv_f["all_converged"], path.name, "OK" if not errors else errors[:5])
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
