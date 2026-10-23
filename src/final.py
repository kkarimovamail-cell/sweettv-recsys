"""Final submission: the configuration that won on validation (MAP@5, cohort A), refit on weeks 11-30.

Selection is read from results_validation.json, nothing is tuned here:
  candidates = each family's selected variant (popularity, PPR, MF, hybrid; personal_history is reference only)
  winner     = highest MAP@5 A; on an exact tie the variant with fewer components wins (hybrid > single method)

Run from the project root:  python -m src.final
"""
import hashlib
import json
import time

from src import config
from src import protocol as P
from src.baselines import popularity_preds
from src.submission import validate_submission, write_submission

CANDIDATES = ["popularity", "ppr", "mf", "hybrid"]
COMPONENTS = {"popularity": 1, "ppr": 1, "mf": 1, "hybrid": 2}
OUT_FILE = config.OUTPUTS_DIR / "results_final.json"

log = config.get_logger("10_final")


def md5(path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def choose(results: dict) -> tuple[dict, list[dict]]:
    rows = [r for r in results["comparison"]["table"]
            if r["selected"] == "yes" and r["family"] in CANDIDATES + ["baseline"]
            and (r["family"] != "baseline" or r["variant"] == "popularity")]
    for r in rows:
        r["family"] = "popularity" if r["family"] == "baseline" else r["family"]
    ranked = sorted(rows, key=lambda r: (-r["map@5_A"], COMPONENTS[r["family"]]))
    return ranked[0], ranked


def predict(family: str, results: dict, train, pool, users):
    m = results["methods"]
    if family == "ppr":
        from src import ppr
        alpha = m["ppr"]["best_alpha"]
        preds, fill, conv = ppr.recommend(ppr.fit_graph(train), pool, users, alpha,
                                          P.popularity_ranking(train, pool))
        return preds, {"alpha": alpha, **{k: v for k, v in m["ppr"]["hyperparameters"].items() if k != "alphas"}}, \
            {"fill": fill, "convergence": conv}
    if family == "mf":
        from src import mf
        cfg = {k: m["mf"]["grid"][m["mf"]["best"]][k] for k in ("factors", "regularization", "alpha")}
        M, users_index, items = mf.user_item_matrix(train)
        model = mf.fit_als(M, **cfg)
        preds, fill = mf.recommend(model.user_factors, model.item_factors, users_index, items, pool, users,
                                   P.popularity_ranking(train, pool))
        return preds, {**cfg, "iterations": mf.ITERATIONS}, {"fill": fill}
    if family == "popularity":
        preds, fill = popularity_preds(train, pool, users)
        return preds, {}, {"fill": fill}
    raise NotImplementedError(f"final refit for family '{family}' is not implemented")


def main():
    t0 = time.time()
    results = json.loads(P.RESULTS_FILE.read_text(encoding=config.ENCODING))
    winner, ranked = choose(results)
    log.info("candidates (MAP@5 A, then fewer components): %s",
             [(r["family"], r["variant"], r["map@5_A"]) for r in ranked])
    log.info("chosen: %s %s", winner["family"], winner["variant"])

    av, items = P.load_airing_views(), P.load_items()
    users = P.submission_users()
    train, pool = P.train_views(av, "test"), P.candidate_pool(items, "test")
    preds, config_used, run_stats = predict(winner["family"], results, train, pool, users)

    path = config.SUBMISSIONS_DIR / "submission_final.csv"
    write_submission(preds, path)
    errors = validate_submission(path, pool)
    digest = md5(path)
    same_as = [p.name for p in sorted(config.SUBMISSIONS_DIR.glob("submission_*.csv"))
               if p.name != path.name and md5(p) == digest]

    final = {
        "method": winner["family"], "variant": winner["variant"], "config": config_used,
        "selection": {"rule": "highest MAP@5 on cohort A among each family's selected variant; "
                              "exact tie -> fewer components",
                      "validation_map@5_A": winner["map@5_A"], "validation_map@5_C": winner["map@5_C"],
                      "candidates": [{k: r[k] for k in ("family", "variant", "map@5_A", "map@5_C")} for r in ranked]},
        "train_weeks": list(P.SPLITS["test"]["train_weeks"]), "pool": "ISO weeks 31-42",
        "pool_size": len(pool), "users": len(users), **run_stats,
        "submission": str(path.relative_to(config.PROJECT_ROOT)), "md5": digest,
        "identical_to": same_as, "validation_errors": errors[:10],
        "seed": config.SEED, "runtime_s": round(time.time() - t0, 2),
    }
    OUT_FILE.write_text(json.dumps(final, ensure_ascii=False, indent=2), encoding=config.ENCODING)
    log.info("%s md5 %s, identical to %s, validation %s, %.2f s",
             path.name, digest, same_as, "OK" if not errors else errors[:5], final["runtime_s"])
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
