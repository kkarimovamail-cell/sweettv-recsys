"""What counts as a training signal for PPR v2, validation only (compared with v2).

Only the training interactions fed to PPR (graph and restart) change; alpha, beta, H, the truth
(screen_time >= 0.8 in weeks 21-30), the pools and the popularity top-up stay as in the protocol.

  threshold T   a (user, airing) pair is a training interaction when screen_time >= T, weight 1 (x recency);
                T = 0.8 is v2
  soft T_min    pairs with screen_time >= T_min, weight = screen_time (x recency)

Run from the project root:  python -m src.signal_exp
"""
import hashlib
import json
import time

import pandas as pd

from src import config
from src import protocol as P
from src.expected import held_out_and_bootstrap
from src.improve import Runner, ap_change
from src.submission import validate_submission, write_submission

VARIANTS = [("threshold", 0.5), ("threshold", 0.65), ("threshold", 0.8),
            ("soft", 0.0), ("soft", 0.1), ("soft", 0.3), ("soft", 0.5)]  # soft 0.0 = every pair with overlap > 0
SELECT_ON = f"map@{P.K}_A"

log = config.get_logger("14_signal")


def load_views() -> pd.DataFrame:
    return pd.read_parquet(config.INTERIM_DIR / "airing_views.parquet", columns=P.AV_COLUMNS + ["screen_time"])


def signal_train(av: pd.DataFrame, split: str, mode: str, t: float) -> pd.DataFrame:
    train = P.train_views(av, split)
    keep = train["screen_time"] >= t
    if mode == "threshold":
        return train.assign(watched=keep)
    if mode == "soft":
        return train.assign(watched=keep, signal=train["screen_time"].where(keep, 0.0))
    raise ValueError(mode)


def make_runner(av, items_df, split, users, alpha, mode, t) -> Runner:
    run = Runner(av, items_df, split, users, alpha)  # popularity top-up and pool from the protocol (0.8)
    run.train = signal_train(av, split, mode, t)     # only PPR's graph and restart see the new signal
    return run


def main():
    results = json.loads(P.RESULTS_FILE.read_text(encoding=config.ENCODING))
    alpha = results["methods"]["ppr"]["best_alpha"]
    v2 = results["improve"]["ppr"]["best"]
    beta, half_life, v2_map = v2["beta"], float(v2["half_life"]), v2[SELECT_ON]

    av, items_df = load_views(), P.load_items()
    truth = P.build_truth(av, "val")
    cohorts = P.build_cohorts(truth, P.submission_users())
    users = sorted(set(cohorts["A"]) | set(cohorts["C"]))
    base_run = Runner(av, items_df, "val", users, alpha)
    base, _, _ = base_run.predict(beta, half_life)  # v2

    prev_v5 = results.get("improve_signal", {}).get("v2_signal", {}).get("final_v5")
    table, all_preds = {}, {}
    for mode, t in VARIANTS:
        t0 = time.time()
        run = make_runner(av, items_df, "val", users, alpha, mode, t)
        preds, fill, conv = run.predict(beta, half_life)
        tr = run.train
        row = {"mode": mode, "t": t, **P.evaluate(preds, truth, cohorts),
               "history_share_A": P.history_share(preds, base_run.train, cohorts["A"]),
               "vs_v2_A": ap_change(preds, base, truth, cohorts["A"]),
               **held_out_and_bootstrap(preds, base, truth, cohorts),
               "identical_to_v2_users": sum(preds[u] == base[u] for u in users),
               "train_pairs": int(tr["watched"].sum()), "train_users": int(tr.loc[tr["watched"], "user_id"].nunique()),
               "train_programmes": int(tr.loc[tr["watched"], "tv_show_id"].nunique()),
               "converged": conv["all_converged"], "fill": fill, "runtime_s": round(time.time() - t0, 2)}
        table[f"{mode}_{t:g}"] = row
        all_preds[f"{mode}_{t:g}"] = preds
        log.info("%-15s A %.6f C %.6f up/down %s dA %s C-A %.6f pairs %d same-as-v2 %d", f"{mode}_{t:g}",
                 row[SELECT_ON], row["map@5_C"], row["vs_v2_A"], row["delta_A_95ci"], row["map@5_C_minus_A"],
                 row["train_pairs"], row["identical_to_v2_users"])

    check = table["threshold_0.8"]
    assert check["identical_to_v2_users"] == len(users) and check[SELECT_ON] == v2_map, "T=0.8 != v2"
    best_name = max(table, key=lambda n: (table[n][SELECT_ON], n == "threshold_0.8"))  # tie -> keep v2
    best = table[best_name]
    accepted = best[SELECT_ON] > v2_map and best["delta_A_95ci"][0] > 0
    out = {"baseline_v2": {"alpha": alpha, "beta": beta, "half_life": v2["half_life"], "threshold": 0.8, SELECT_ON: v2_map},
           "definition": __doc__.split("Run from")[0].strip(), "table": table,
           "check_T0.8_equals_v2": {"users_identical": check["identical_to_v2_users"], "users": len(users)},
           "best": best_name, "rule": "accept only if MAP@5 A > v2 and the 95% CI of the A gain excludes 0",
           "accepted": accepted}
    log.info("best %s accepted %s", best_name, accepted)

    # an existing v5 is replaced only by a different variant that beats it with a CI above 0
    rebuild = accepted
    if prev_v5 is not None:
        v5_name = f"{prev_v5['mode']}_{prev_v5['t']:g}"
        v5_preds = all_preds[v5_name]
        for name, row in table.items():
            boot = held_out_and_bootstrap(all_preds[name], v5_preds, truth, cohorts)
            row["vs_v5_A"] = ap_change(all_preds[name], v5_preds, truth, cohorts["A"])
            row["vs_v5_delta_A_mean"], row["vs_v5_delta_A_95ci"] = boot["delta_A_mean"], boot["delta_A_95ci"]
            log.info("%-15s vs v5 (%s): up/down %s dA %s", name, v5_name, row["vs_v5_A"], row["vs_v5_delta_A_95ci"])
        rebuild = (best_name != v5_name and best[SELECT_ON] > table[v5_name][SELECT_ON]
                   and best["vs_v5_delta_A_95ci"][0] > 0)
        out.update({"current_v5": v5_name, "v5_rule": "rebuild only if the best variant differs from the current v5, "
                    "beats it on MAP@5 A and the 95% CI of the A gain over v5 excludes 0", "v5_rebuilt": rebuild})
        if not rebuild:
            out["final_v5"] = prev_v5  # file left untouched
        log.info("current v5 %s, rebuild %s", v5_name, rebuild)

    if rebuild:
        t1 = time.time()
        sub_users = P.submission_users()
        run_f = make_runner(av, items_df, "test", sub_users, alpha, best["mode"], best["t"])
        preds_f, fill_f, conv_f = run_f.predict(beta, half_life)
        path = config.SUBMISSIONS_DIR / "submission_v5.csv"
        write_submission(preds_f, path)
        errors = validate_submission(path, run_f.pool)
        out["final_v5"] = {"alpha": alpha, "beta": beta, "half_life": v2["half_life"], "mode": best["mode"], "t": best["t"],
                           "train_weeks": list(P.SPLITS["test"]["train_weeks"]), "fill": fill_f,
                           "converged": conv_f["all_converged"], "submission": path.name,
                           "md5": hashlib.md5(path.read_bytes()).hexdigest(),
                           "validation_errors": errors[:10], "runtime_s": round(time.time() - t1, 2)}
        log.info("v5: %s", out["final_v5"])
        if errors:
            raise SystemExit(1)
    P.update_results("improve_signal", "v2_signal", out)


if __name__ == "__main__":
    main()
