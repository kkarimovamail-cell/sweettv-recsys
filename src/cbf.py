"""Content-based filtering on programme metadata, on the shared protocol; then a hybrid with PPR v2.

  item vector  [one-hot category | multi-hot genres 1-3 (one vocabulary) | TF-IDF directors | TF-IDF actors];
               each block L2-normalised (an empty block stays zero), then the whole vector L2-normalised.
               TF-IDF token = one person (actors split on ","), tf = 1, idf = ln((1 + N) / (1 + df)) + 1
               over all N programmes in items.parquet (same formula as scikit-learn's smooth_idf).
  profile      sum_i w_ui x_i / sum_i w_ui, w_ui = recency-weighted watched airings in the train window
               (0.5 ** (delta / H), H as in v2)
  score        cosine(profile, x_i) over pool programmes; score > 0 kept, ties -> smaller id, popularity top-up
  hybrid       candidates = v2 top-50 followed by CBF-only programmes of the CBF top-50;
               score = v2 / max_u(v2) + mu * cbf / max_u(cbf) (0 for a list a programme is not in);
               zero scores dropped, remaining ties keep that candidate order; mu = 0 must reproduce v2

Run from the project root:  python -m src.cbf
"""
import hashlib
import json
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp

from src import config, ppr
from src import protocol as P
from src.expected import candidates, held_out_and_bootstrap
from src.improve import Runner, ap_change
from src.submission import validate_submission, write_submission

GENRE_COLS = ["tv_show_genre_1", "tv_show_genre_2", "tv_show_genre_3"]
MUS = [0.0, 0.05, 0.1, 0.25]
DEPTH = 50
BATCH_USERS = 1024
SELECT_ON = f"map@{P.K}_A"

log = config.get_logger("13_cbf")


# --- item vectors -------------------------------------------------------------
def token_block(tokens: list[list[str]], idf: bool) -> sp.csr_matrix:
    vocab = {t: k for k, t in enumerate(sorted({t for ts in tokens for t in ts}))}
    rows = [r for r, ts in enumerate(tokens) for _ in ts]
    cols = [vocab[t] for ts in tokens for t in ts]
    X = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(tokens), len(vocab)))
    if idf:
        df = np.asarray((X > 0).sum(axis=0)).ravel()
        X = X @ sp.diags(np.log((1 + len(tokens)) / (1 + df)) + 1)
    return l2_rows(X.tocsr())


def l2_rows(X: sp.csr_matrix) -> sp.csr_matrix:
    norm = np.sqrt(np.asarray(X.multiply(X).sum(axis=1)).ravel())
    return (sp.diags(np.divide(1.0, norm, out=np.zeros_like(norm), where=norm > 0)) @ X).tocsr()


def split_people(value) -> list[str]:
    return [] if pd.isna(value) else sorted({p.strip() for p in str(value).split(",") if p.strip()})


def item_matrix(items_df: pd.DataFrame) -> tuple[sp.csr_matrix, np.ndarray, dict]:
    df = items_df.assign(id_num=items_df["tv_show_id"].astype("int64")).sort_values("id_num")
    blocks = {
        "category": token_block([[c] if pd.notna(c) else [] for c in df["tv_show_category"]], idf=False),
        "genres": token_block([sorted({g for g in row if pd.notna(g)}) for row in df[GENRE_COLS].itertuples(index=False)], idf=False),
        "directors": token_block([split_people(d) for d in df["director"]], idf=True),
        "actors": token_block([split_people(a) for a in df["actors"]], idf=True),
    }
    V = l2_rows(sp.hstack(list(blocks.values())).tocsr())
    stats = {name: {"features": b.shape[1], "programmes_with_any": int((b.getnnz(axis=1) > 0).sum())}
             for name, b in blocks.items()}
    return V, df["tv_show_id"].to_numpy(), stats


# --- recommendations ---------------------------------------------------------
def user_profiles(V: sp.csr_matrix, ids: np.ndarray, train: pd.DataFrame, half_life: float, end_week: int):
    """Unit-length weighted-mean profiles (rows aligned with the returned user index)."""
    _, H, train_users, train_items = ppr.interaction_matrices(train, half_life, end_week)
    Vt = V[pd.Index(ids).get_indexer(train_items)]
    prof = l2_rows(sp.csr_matrix(H @ Vt / np.asarray(H.sum(axis=1))))  # weighted mean, then unit length
    return prof, pd.Index(train_users)


def rank_users(V: sp.csr_matrix, ids: np.ndarray, train: pd.DataFrame, half_life: float, end_week: int,
               pool: set[str], users: list[str], depth: int = P.K):
    """{user: ids}, {user: cosine scores} for known users; pool ids sorted by number so ties -> smaller id."""
    prof, train_users = user_profiles(V, ids, train, half_life, end_week)
    pool_rows = np.flatnonzero(np.isin(ids, list(pool)))
    Vp = V[pool_rows].T.tocsc()
    user_pos = train_users.get_indexer(users)
    known = [(u, p) for u, p in zip(users, user_pos) if p >= 0]
    ranked, scores = {}, {}
    for b in range(0, len(known), BATCH_USERS):
        batch = known[b:b + BATCH_USERS]
        S = (prof[[p for _, p in batch]] @ Vp).toarray()
        order = np.argsort(-S, axis=1, kind="stable")[:, :depth]
        for j, (u, _) in enumerate(batch):
            top = order[j][S[j, order[j]] > 0]
            ranked[u] = ids[pool_rows[top]].tolist()
            scores[u] = S[j, top]
    return ranked, scores


def fuse(v2_ranked, v2_scores, cbf_ranked, cbf_scores, mu: float) -> dict:
    out = {}
    for u in set(v2_ranked) | set(cbf_ranked):
        a_ids, b_ids = v2_ranked.get(u, []), cbf_ranked.get(u, [])
        a = dict(zip(a_ids, v2_scores[u] / v2_scores[u].max())) if a_ids else {}
        b = dict(zip(b_ids, cbf_scores[u] / cbf_scores[u].max())) if b_ids else {}
        cand = a_ids + [i for i in b_ids if i not in a]
        score = np.array([a.get(i, 0.0) + mu * b.get(i, 0.0) for i in cand])
        order = np.argsort(-score, kind="stable")
        out[u] = [cand[k] for k in order if score[k] > 0][:P.K]
    return out


def novelty(preds: dict, train: pd.DataFrame, truth: dict, users: list[str]) -> dict:
    seen = set(zip(train.loc[train["watched"], "user_id"], train.loc[train["watched"], "tv_show_id"]))
    new = [(u, i) for u in users for i in preds[u] if (u, i) not in seen]
    hits = sum(i in truth[u] for u, i in new)
    return {"repeat_share_A": round(1 - len(new) / (P.K * len(users)), 4),
            "new_share_A": round(len(new) / (P.K * len(users)), 4),
            "new_slots_A": len(new), "new_slots_in_truth_A": hits,
            "new_slot_hit_rate_A": round(hits / len(new), 4) if new else None}


# --- main --------------------------------------------------------------------
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
    end_week = P.SPLITS["val"]["train_weeks"][1]

    t0 = time.time()
    V, ids, vec_stats = item_matrix(items_df)
    cbf_ranked, cbf_scores = rank_users(V, ids, run.train, half_life, end_week, run.pool, users, depth=DEPTH)
    cbf_preds, cbf_fill = P.fill_to_k(cbf_ranked, users, run.popular, run.pool)
    cbf_s = round(time.time() - t0, 2)
    base, _, _ = run.predict(beta, half_life)  # v2 on validation

    cbf_row = {**P.evaluate(cbf_preds, truth, cohorts), "history_share_A": P.history_share(cbf_preds, run.train, cohorts["A"]),
               **novelty(cbf_preds, run.train, truth, cohorts["A"]), "fill": cbf_fill, "runtime_s": cbf_s}
    v2_ref = {**P.evaluate(base, truth, cohorts), **novelty(base, run.train, truth, cohorts["A"])}
    log.info("CBF %s", {k: cbf_row[k] for k in (SELECT_ON, "map@5_C", "repeat_share_A", "new_share_A", "new_slots_in_truth_A")})
    log.info("v2 reference %s", v2_ref)
    P.save_preds(cbf_preds, config.INTERIM_DIR / "preds_val_cbf.parquet")
    P.update_results("methods", "cbf", {
        **cbf_row, "v2_reference": v2_ref, "vectors": {"programmes": V.shape[0], "features": V.shape[1], "blocks": vec_stats},
        "hyperparameters": {"half_life": v2["half_life"], "candidate_depth_for_hybrid": DEPTH,
                            "definition": __doc__.split("hybrid ")[0].strip()}, "seed": config.SEED})

    # hybrid with the best current variant (v2)
    _, v2_ranked, v2_scores = candidates(run, beta, half_life)
    table = {}
    for mu in MUS:
        preds, fill = P.fill_to_k(fuse(v2_ranked, v2_scores, cbf_ranked, cbf_scores, mu), users, run.popular, run.pool)
        row = {**P.evaluate(preds, truth, cohorts), **novelty(preds, run.train, truth, cohorts["A"]),
               "vs_v2_A": ap_change(preds, base, truth, cohorts["A"]),
               **held_out_and_bootstrap(preds, base, truth, cohorts),
               "identical_to_v2_users": sum(preds[u] == base[u] for u in users), "fill": fill}
        table[f"mu={mu:g}"] = row
        log.info("mu %-5g A %.6f C %.6f up/down %s dA %s C-A %.6f new %.4f same-as-v2 %d", mu, row[SELECT_ON], row["map@5_C"],
                 row["vs_v2_A"], row["delta_A_95ci"], row["map@5_C_minus_A"], row["new_share_A"], row["identical_to_v2_users"])
    assert table["mu=0"]["identical_to_v2_users"] == len(users) and table["mu=0"][SELECT_ON] == v2_map, "mu=0 != v2"

    best_name = max(table, key=lambda n: table[n][SELECT_ON])  # mu=0 first: ties keep v2
    improved = table[best_name][SELECT_ON] > v2_map
    out = {"baseline_v2": {"alpha": alpha, "beta": beta, "half_life": v2["half_life"], SELECT_ON: v2_map},
           "hyperparameters": {"mus": MUS, "candidate_depth": DEPTH, "normalisation": "score / user's max, per list"},
           "table": table, "check_mu0_equals_v2": {"users_identical": table["mu=0"]["identical_to_v2_users"], "users": len(users)},
           "best": best_name, "improved": improved}

    if improved:
        t1 = time.time()
        mu = float(best_name.split("=")[1])
        sub_users = P.submission_users()
        run_f = Runner(av, items_df, "test", sub_users, alpha)
        _, r_f, s_f = candidates(run_f, beta, half_life)
        c_f, cs_f = rank_users(V, ids, run_f.train, half_life, P.SPLITS["test"]["train_weeks"][1], run_f.pool, sub_users, DEPTH)
        preds_f, fill_f = P.fill_to_k(fuse(r_f, s_f, c_f, cs_f, mu), sub_users, run_f.popular, run_f.pool)
        path = config.SUBMISSIONS_DIR / "submission_v4.csv"
        write_submission(preds_f, path)
        errors = validate_submission(path, run_f.pool)
        out["final_v4"] = {"alpha": alpha, "beta": beta, "half_life": v2["half_life"], "mu": mu,
                           "train_weeks": list(P.SPLITS["test"]["train_weeks"]), "fill": fill_f,
                           "submission": path.name, "md5": hashlib.md5(path.read_bytes()).hexdigest(),
                           "validation_errors": errors[:10], "runtime_s": round(time.time() - t1, 2)}
        log.info("v4: %s", out["final_v4"])
        if errors:
            raise SystemExit(1)
    P.update_results("improve_cbf", "v2_plus_cbf", out)
    log.info("best %s improved %s", best_name, improved)


if __name__ == "__main__":
    main()
