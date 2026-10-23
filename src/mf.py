"""Matrix factorization for implicit feedback: ALS from `implicit`, on the shared protocol.

  matrix   user x item = number of watched airings in the train window, all users of the window
  model    implicit.als.AlternatingLeastSquares on CPU, num_threads=1, random_state=SEED;
           implicit 0.7.3 multiplies the matrix by `alpha` before fitting (implicit/cpu/als.py)
  scores   user_factors @ item_factors.T over pool programmes that have a factor (watched in train);
           already-watched programmes are kept (the library's recommend() is not used)
  output   top-5 by score (ties: smaller id), cold-start users topped up from popularity

Run from the project root:  python -m src.mf
"""
import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")  # numpy here is built on Apple Accelerate

import itertools  # noqa: E402
import time  # noqa: E402

import implicit  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scipy.sparse as sp  # noqa: E402
from implicit.als import AlternatingLeastSquares  # noqa: E402

from src import config  # noqa: E402
from src import protocol as P  # noqa: E402
from src.submission import validate_submission, write_submission  # noqa: E402

GRID = {"factors": [64], "regularization": [0.01, 0.1], "alpha": [10.0, 40.0]}
# diagnostics outside the grid (issues_log #26): reported separately, never used for selection
OFF_GRID = {"counts_f64_reg0.01_a1": ("counts", 1.0), "binary_f64_reg0.01_a10": ("binary", 10.0),
            "log1p_f64_reg0.01_a10": ("log1p", 10.0)}
OFF_GRID_NOTE = "off-grid, not used for selection (вне сетки, для выбора не использовались)"
ITERATIONS = 15
BATCH_USERS = 2048
SELECT_ON = f"map@{P.K}_A"

log = config.get_logger("08_mf")


def user_item_matrix(train: pd.DataFrame):
    """CSR of watched-airing counts; users sorted, items sorted by numeric id."""
    counts = P.watched_counts(train)
    items = sorted(counts["tv_show_id"].unique(), key=int)
    users = sorted(counts["user_id"].unique())
    r = pd.Index(users).get_indexer(counts["user_id"])
    c = pd.Index(items).get_indexer(counts["tv_show_id"])
    M = sp.csr_matrix((counts["n_watched"].to_numpy(dtype=np.float32), (r, c)), shape=(len(users), len(items)))
    return M, pd.Index(users), np.array(items)


def fit_als(M: sp.csr_matrix, factors: int, regularization: float, alpha: float) -> AlternatingLeastSquares:
    model = AlternatingLeastSquares(factors=factors, regularization=regularization, alpha=alpha,
                                    iterations=ITERATIONS, use_gpu=False, num_threads=1,
                                    random_state=config.SEED)
    model.fit(M, show_progress=False)
    return model


def transform_matrix(M: sp.csr_matrix, how: str) -> sp.csr_matrix:
    out = M.copy()
    if how == "binary":
        out.data[:] = 1.0
    elif how == "log1p":
        out.data = np.log1p(out.data).astype(np.float32)
    elif how != "counts":
        raise ValueError(how)
    return out


def rank_users(user_factors, item_factors, users_index: pd.Index, items: np.ndarray,
               pool: set[str], users: list[str], depth: int = P.K) -> dict[str, list[str]]:
    """Top `depth` pool programmes with a factor per known user, by score (ties: smaller id)."""
    pool_idx = np.flatnonzero(np.isin(items, list(pool)))
    pool_factors = item_factors[pool_idx]
    known = [(u, p) for u, p in zip(users, users_index.get_indexer(users)) if p >= 0]
    ranked = {}
    for b in range(0, len(known), BATCH_USERS):
        batch = known[b:b + BATCH_USERS]
        S = user_factors[[p for _, p in batch]] @ pool_factors.T
        order = np.argsort(-S, axis=1, kind="stable")[:, :depth]  # stable -> ties keep ascending id order
        for j, (u, _) in enumerate(batch):
            ranked[u] = items[pool_idx[order[j]]].tolist()
    return ranked


def recommend(user_factors, item_factors, users_index: pd.Index, items: np.ndarray,
              pool: set[str], users: list[str], popular: list[str]):
    ranked = rank_users(user_factors, item_factors, users_index, items, pool, users)
    return P.fill_to_k(ranked, users, popular, pool)


def blas_name() -> str:
    return np.show_config(mode="dicts")["Build Dependencies"]["blas"]["name"]


def main():
    env = {"implicit": implicit.__version__, "numpy_blas": blas_name(),
           "OPENBLAS_NUM_THREADS": os.environ["OPENBLAS_NUM_THREADS"],
           "VECLIB_MAXIMUM_THREADS": os.environ["VECLIB_MAXIMUM_THREADS"], "num_threads": 1, "gpu": False}
    log.info("environment: %s", env)

    av, items_df = P.load_airing_views(), P.load_items()
    truth = P.build_truth(av, "val")
    cohorts = P.build_cohorts(truth, P.submission_users())
    users = sorted(set(cohorts["A"]) | set(cohorts["C"]))
    train, pool = P.train_views(av, "val"), P.candidate_pool(items_df, "val")
    M, users_index, items = user_item_matrix(train)
    popular = P.popularity_ranking(train, pool)
    log.info("val matrix: %d users x %d items, %d non-zeros; pool items with a factor: %d of %d",
             M.shape[0], M.shape[1], M.nnz, int(np.isin(items, list(pool)).sum()), len(pool))

    grid, models = {}, {}
    for factors, reg, alpha in itertools.product(*GRID.values()):
        key = f"f{factors}_reg{reg}_a{alpha:g}"
        t0 = time.time()
        model = fit_als(M, factors, reg, alpha)
        fit_s = time.time() - t0
        preds, fill = recommend(model.user_factors, model.item_factors, users_index, items, pool, users, popular)
        scores = P.evaluate(preds, truth, cohorts)
        scores["history_share_A"] = P.history_share(preds, train, cohorts["A"])
        grid[key] = {**scores, "factors": factors, "regularization": reg, "alpha": alpha,
                     "fill": fill, "fit_s": round(fit_s, 2), "runtime_s": round(time.time() - t0, 2)}
        models[key] = (model, preds)
        P.save_preds(preds, config.INTERIM_DIR / f"preds_val_mf_{key}.parquet")
        log.info("%-22s %s  fit %.2f s", key, scores, fit_s)

    best = max(grid, key=lambda k: grid[k][SELECT_ON])  # first of equal scores in grid order
    best_cfg = {k: grid[best][k] for k in ("factors", "regularization", "alpha")}
    model, preds = models[best]

    # reproducibility: refit the best configuration from scratch and compare
    again = fit_als(M, **best_cfg)
    preds_again, _ = recommend(again.user_factors, again.item_factors, users_index, items, pool, users, popular)
    repro = {
        "user_factors_identical": bool(np.array_equal(model.user_factors, again.user_factors)),
        "item_factors_identical": bool(np.array_equal(model.item_factors, again.item_factors)),
        "max_abs_factor_diff": float(max(np.abs(model.user_factors - again.user_factors).max(),
                                         np.abs(model.item_factors - again.item_factors).max())),
        "predictions_identical": preds == preds_again,
        "map_again": P.evaluate(preds_again, truth, cohorts),
    }
    log.info("best %s; reproducibility %s", best, repro)

    off_grid = {}
    for key, (how, alpha) in OFF_GRID.items():  # after selection on purpose: these cannot influence it
        t0 = time.time()
        diag = fit_als(transform_matrix(M, how), factors=64, regularization=0.01, alpha=alpha)
        fit_s = time.time() - t0
        d_preds, d_fill = recommend(diag.user_factors, diag.item_factors, users_index, items, pool, users, popular)
        d_scores = P.evaluate(d_preds, truth, cohorts)
        d_scores["history_share_A"] = P.history_share(d_preds, train, cohorts["A"])
        off_grid[key] = {**d_scores, "matrix_transform": how, "factors": 64, "regularization": 0.01,
                         "alpha": alpha, "fill": d_fill, "fit_s": round(fit_s, 2),
                         "runtime_s": round(time.time() - t0, 2), "note": OFF_GRID_NOTE}
        P.save_preds(d_preds, config.INTERIM_DIR / f"preds_val_mf_{key}.parquet")
        log.info("off-grid %-24s %s", key, d_scores)

    P.save_preds(preds, config.INTERIM_DIR / "preds_val_mf.parquet")
    P.update_results("methods", "mf", {
        **{k: grid[best][k] for k in (f"map@{P.K}_A", f"map@{P.K}_C", "history_share_A")},
        "best": best, "selected_on": SELECT_ON, "grid": grid, "reproducibility": repro,
        "off_grid": {"note": OFF_GRID_NOTE, "variants": off_grid},
        "hyperparameters": {**GRID, "iterations": ITERATIONS, "batch_users": BATCH_USERS,
                            "matrix": "watched-airing counts per (user, programme)"},
        "matrix": {"users": M.shape[0], "items": M.shape[1], "nnz": int(M.nnz)},
        "environment": env, "seed": config.SEED,
    })

    # final: train on weeks 11-30, predict for submission users, pool of weeks 31-42
    t1 = time.time()
    train_f, pool_f = P.train_views(av, "test"), P.candidate_pool(items_df, "test")
    M_f, users_f, items_f = user_item_matrix(train_f)
    model_f = fit_als(M_f, **best_cfg)
    preds_f, fill_f = recommend(model_f.user_factors, model_f.item_factors, users_f, items_f, pool_f,
                                P.submission_users(), P.popularity_ranking(train_f, pool_f))
    path = config.SUBMISSIONS_DIR / "submission_mf.csv"
    write_submission(preds_f, path)
    errors = validate_submission(path, pool_f)
    P.update_results("final", "mf", {**best_cfg, "train_weeks": list(P.SPLITS["test"]["train_weeks"]),
                                     "matrix": {"users": M_f.shape[0], "items": M_f.shape[1], "nnz": int(M_f.nnz)},
                                     "fill": fill_f, "submission": path.name, "validation_errors": errors[:10],
                                     "runtime_s": round(time.time() - t1, 2)})
    log.info("final matrix %s; fill %s; %s: %s", M_f.shape, fill_f, path.name, "OK" if not errors else errors[:5])
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
