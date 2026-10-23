"""Shared evaluation protocol: windows, ground truth, cohorts, candidate pools, popularity fill.

  validation: train on ISO weeks 11-20, ground truth from weeks 21-30
  final:      train on weeks 11-30, predict weeks 31-42
  truth:      a user's top-5 programmes by number of watched airings in the target weeks
              (ties: larger summed overlap_s of those watched airings, then smaller numeric id)
  cohort A:   submission users with >= 5 programmes in the truth
  cohort C:   all log users with >= 5 (sampled down to 20,000 with seed 42 if larger)
  pool:       tv_show_id != 0 airing in the target window's ISO weeks; repeats of watched programmes allowed
  fill:       cold-start users and lists shorter than 5 are topped up from popularity -> exactly 5 distinct ids

Run from the project root:  python -m src.protocol   (prints the data accounting, saves it to results_validation.json)
"""
import json

import numpy as np
import pandas as pd

from src import config
from src.metrics import mapk

K = config.K
SPLITS = {
    "val": {"train_weeks": (11, 20), "target_weeks": (21, 30), "pool_col": "n_airings_w21_30"},
    "test": {"train_weeks": (11, 30), "target_weeks": (31, 42), "pool_col": "n_airings_w31_42"},
}
COHORT_C_MAX = 20_000
GAP_WEEKS = (25, 27)  # weeks with a collapse in log volume (issues_log #14)
RESULTS_FILE = config.OUTPUTS_DIR / "results_validation.json"
AV_COLUMNS = ["user_id", "tv_show_id", "overlap_s", "watched", "week"]

log = config.get_logger("04_protocol")


# --- loading -----------------------------------------------------------------
def load_airing_views() -> pd.DataFrame:
    return pd.read_parquet(config.INTERIM_DIR / "airing_views.parquet", columns=AV_COLUMNS)


def load_items() -> pd.DataFrame:
    return pd.read_parquet(config.INTERIM_DIR / "items.parquet")


def submission_users() -> list[str]:
    """User ids in the template's order."""
    return pd.read_csv(config.SUBMISSION_TEMPLATE, encoding=config.ENCODING, dtype=str)["user_id"].tolist()


def in_weeks(av: pd.DataFrame, weeks: tuple[int, int]) -> pd.DataFrame:
    return av[av["week"].between(*weeks)]


def train_views(av: pd.DataFrame, split: str) -> pd.DataFrame:
    return in_weeks(av, SPLITS[split]["train_weeks"])


def candidate_pool(items: pd.DataFrame, split: str) -> set[str]:
    return set(items.loc[items[SPLITS[split]["pool_col"]] > 0, "tv_show_id"])


# --- ranking helpers ---------------------------------------------------------
def watched_counts(av: pd.DataFrame) -> pd.DataFrame:
    """Per (user, programme): number of watched airings and their summed overlap."""
    w = av[av["watched"]]
    return w.groupby(["user_id", "tv_show_id"], as_index=False).agg(
        n_watched=("watched", "size"), overlap_s=("overlap_s", "sum"))


def rank_per_user(counts: pd.DataFrame, k: int | None = None) -> pd.DataFrame:
    """Sort by n_watched desc, overlap_s desc, numeric id asc; add 1-based 'rank'."""
    c = counts.assign(id_num=counts["tv_show_id"].astype("int64"))
    c = c.sort_values(["user_id", "n_watched", "overlap_s", "id_num"], ascending=[True, False, False, True])
    c["rank"] = c.groupby("user_id").cumcount() + 1
    if k is not None:
        c = c[c["rank"] <= k]
    return c.drop(columns="id_num").reset_index(drop=True)


def to_lists(ranked: pd.DataFrame) -> dict[str, list[str]]:
    return ranked.groupby("user_id", sort=False)["tv_show_id"].agg(list).to_dict()


def build_truth(av: pd.DataFrame, split: str = "val", exclude_weeks: tuple[int, int] | None = None) -> dict[str, list[str]]:
    target = in_weeks(av, SPLITS[split]["target_weeks"])
    if exclude_weeks is not None:
        target = target[~target["week"].between(*exclude_weeks)]
    counts = watched_counts(target)
    n_prog = counts.groupby("user_id").size()
    counts = counts[counts["user_id"].isin(n_prog[n_prog >= K].index)]
    return to_lists(rank_per_user(counts, K))


def build_cohorts(truth: dict, sub_users: list[str]) -> dict[str, list[str]]:
    cohort_a = [u for u in sub_users if u in truth]
    cohort_c = sorted(truth)
    if len(cohort_c) > COHORT_C_MAX:
        rng = np.random.default_rng(config.SEED)
        cohort_c = sorted(rng.choice(cohort_c, COHORT_C_MAX, replace=False).tolist())
    return {"A": cohort_a, "C": cohort_c}


def popularity_ranking(train: pd.DataFrame, pool: set[str]) -> list[str]:
    """Pool programmes by number of distinct users with a watched airing in train (ties: smaller id)."""
    w = train[train["watched"] & train["tv_show_id"].isin(pool)]
    n_users = w.groupby("tv_show_id")["user_id"].nunique().rename("n_users").reset_index()
    n_users["id_num"] = n_users["tv_show_id"].astype("int64")
    return n_users.sort_values(["n_users", "id_num"], ascending=[False, True])["tv_show_id"].tolist()


def fill_to_k(ranked: dict[str, list[str]], users: list[str], popular: list[str], pool: set[str]) -> tuple[dict, dict]:
    """Keep a method's in-pool, de-duplicated ids (in order), top up from popularity to exactly K."""
    preds, n_empty, n_short, n_filled = {}, 0, 0, 0
    for u in users:
        own = list(dict.fromkeys(i for i in ranked.get(u, []) if i in pool))[:K]
        n_empty += not own
        n_short += len(own) < K
        n_filled += K - len(own)
        seen = set(own)
        for p in popular:
            if len(own) == K:
                break
            if p not in seen:
                own.append(p)
                seen.add(p)
        assert len(own) == K, f"cannot fill {u} to {K} ids"
        preds[u] = own
    return preds, {"users": len(users), "users_without_own_ids": n_empty,
                   "users_topped_up": n_short, "slots_from_popularity": n_filled}


# --- evaluation and storage --------------------------------------------------
def evaluate(preds: dict, truth: dict, cohorts: dict) -> dict:
    return {f"map@{K}_{name}": round(mapk([truth[u] for u in users], [preds[u] for u in users], K), 6)
            for name, users in cohorts.items()}


def history_share(preds: dict, train: pd.DataFrame, users: list[str]) -> float:
    """Share of recommended slots that the user already watched in the train window."""
    seen = set(zip(train.loc[train["watched"], "user_id"], train.loc[train["watched"], "tv_show_id"]))
    hits = sum((u, i) in seen for u in users for i in preds[u])
    return round(hits / (K * len(users)), 4)


def save_preds(preds: dict, path) -> None:
    rows = [(u, r + 1, i) for u, ids in preds.items() for r, i in enumerate(ids)]
    df = pd.DataFrame(rows, columns=["user_id", "rank", "tv_show_id"])
    df["rank"] = df["rank"].astype("int8")
    df.to_parquet(path, index=False)


def update_results(section: str, key: str, value) -> None:
    results = json.loads(RESULTS_FILE.read_text(encoding=config.ENCODING)) if RESULTS_FILE.exists() else {}
    results.setdefault(section, {})[key] = value
    RESULTS_FILE.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding=config.ENCODING)


# --- data accounting ---------------------------------------------------------
def no_watched(users, train: pd.DataFrame) -> int:
    return len(set(users) - set(train.loc[train["watched"], "user_id"]))


def gap_accounting(av: pd.DataFrame, truth: dict, cohorts: dict) -> dict:
    target = in_weeks(av, SPLITS["val"]["target_weeks"])
    w = target[target["watched"]]
    in_gap = w["week"].between(*GAP_WEEKS)
    out = {"gap_weeks": list(GAP_WEEKS),
           "watched_airings_weeks_21_30": len(w), "of_them_in_gap_weeks": int(in_gap.sum())}
    truth_pairs = pd.DataFrame([(u, i) for u, ids in truth.items() for i in ids], columns=["user_id", "tv_show_id"])
    wt = w.merge(truth_pairs, on=["user_id", "tv_show_id"])
    truth_no_gap = build_truth(av, "val", exclude_weeks=GAP_WEEKS)
    for name, users in cohorts.items():
        part = wt[wt["user_id"].isin(users)]
        out[f"cohort_{name}"] = {
            "truth_watched_airings": len(part),
            "truth_watched_airings_in_gap_weeks": int(part["week"].between(*GAP_WEEKS).sum()),
            "would_lose_5plus_without_gap_weeks": sum(u not in truth_no_gap for u in users),
        }
    return out


def data_accounting(av: pd.DataFrame, items: pd.DataFrame) -> dict:
    sub = submission_users()
    truth = build_truth(av, "val")
    cohorts = build_cohorts(truth, sub)
    target_counts = watched_counts(in_weeks(av, SPLITS["val"]["target_weeks"]))
    n_prog = target_counts.groupby("user_id").size()
    sub_n = pd.Series(sub).map(n_prog).fillna(0)
    pools = {s: candidate_pool(items, s) for s in SPLITS}
    assert all(i in pools["val"] for ids in truth.values() for i in ids), "truth item outside the val pool"

    acc = {
        "cohort_sizes": {"A": len(cohorts["A"]), "C": len(cohorts["C"]),
                         "C_sampled": len(truth) > COHORT_C_MAX, "log_users_with_5plus": len(truth)},
        "submission_users_by_truth_size": {"total": len(sub), "5plus": int((sub_n >= K).sum()),
                                           "1_to_4": int(sub_n.between(1, K - 1).sum()), "0": int((sub_n == 0).sum())},
        "cold_start_no_watched_in_train": {
            "val_cohort_A": no_watched(cohorts["A"], train_views(av, "val")),
            "val_cohort_C": no_watched(cohorts["C"], train_views(av, "val")),
            "test_submission_users": no_watched(sub, train_views(av, "test")),
        },
        "pool_sizes": {
            s: {"programmes": len(p),
                "watched_in_train": len(p & set(train_views(av, s).loc[lambda d: d["watched"], "tv_show_id"]))}
            for s, p in pools.items()},
        "gap_weeks_25_27": gap_accounting(av, truth, cohorts),
    }
    return acc


def main():
    av, items = load_airing_views(), load_items()
    acc = data_accounting(av, items)
    log.info("data accounting:\n%s", json.dumps(acc, ensure_ascii=False, indent=2))
    update_results("protocol", "description", __doc__.split("Run from")[0].strip())
    update_results("protocol", "data_accounting", acc)


if __name__ == "__main__":
    main()
