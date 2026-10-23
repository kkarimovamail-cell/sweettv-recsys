"""Viewing sessions -> (user, airing) interactions with screen_time.

Rule (mirrors how the Kaggle answer was built):
  airing       = schedule row with tv_show_id != "0", end = start + duration
  sessions     = overlapping sessions of one user on one channel are merged first
  overlap      = |session ∩ airing| on the same channel, summed per (user, airing), clipped to duration
  screen_time  = overlap / duration;  watched = screen_time >= WATCH_THRESHOLD
Times come from start_time/stop_time, never from `duraton` (117 rows are off by the DST hour).

Run from the project root:  python -m src.join
Writes outputs/interim/airing_views.parquet, outputs/interim/items.parquet, outputs/join_stats.json
"""
import json
import time

import numpy as np
import pandas as pd

from src import config

CHANNEL_KEY = 10**10  # channel_id * CHANNEL_KEY + unix seconds keeps every channel in its own range
ITEM_ATTRS = ["tv_show_title", "tv_show_category", "tv_show_genre_1", "tv_show_genre_2", "tv_show_genre_3",
              "director", "actors"]
ITEM_WINDOWS = {"w11_20": (11, 20), "w21_30": (21, 30), "w31_42": (31, 42)}
THRESHOLDS = [0.5, 0.8, 0.9]
HIST_BINS = np.round(np.linspace(0, 1, 11), 1)

log = config.get_logger("02_join")


def to_seconds(s: pd.Series) -> np.ndarray:
    return s.dt.as_unit("s").astype("int64").to_numpy()


def canonical_ids(schedule: pd.DataFrame) -> pd.DataFrame:
    """tv_show_id -> str(int(id)): drops leading zeros (e.g. 00078499 -> 78499); fails on a collision."""
    raw = schedule["tv_show_id"]
    canon = raw.astype("int64").astype(str)
    pairs = pd.DataFrame({"raw": raw, "canon": canon}).drop_duplicates()
    assert not pairs["canon"].duplicated().any(), "two raw ids map to one canonical id"
    return schedule.assign(tv_show_id=canon)


def dedup_schedule(schedules: list[pd.DataFrame]) -> tuple[pd.DataFrame, int]:
    """Concatenate schedule files and drop repeated (channel_id, start_time) rows at the seams."""
    sched = pd.concat(schedules, ignore_index=True)
    deduped = sched.drop_duplicates(["channel_id", "start_time"], keep="first", ignore_index=True)
    return deduped, len(sched) - len(deduped)


def build_airings(schedule: pd.DataFrame) -> pd.DataFrame:
    """Labelled schedule rows sorted by (channel, start); airing_id = row position."""
    air = schedule.loc[schedule["tv_show_id"] != "0", ["channel_id", "start_time", "duration", "tv_show_id"]]
    air = air.sort_values(["channel_id", "start_time"], kind="stable", ignore_index=True)
    assert (air["duration"] > 0).all(), "labelled airing with non-positive duration"
    air["start_s"] = to_seconds(air["start_time"])
    air["end_s"] = air["start_s"] + air["duration"].to_numpy()
    return air


def merge_sessions(views: pd.DataFrame, shift_s: int = 0) -> pd.DataFrame:
    """Merge sessions of one user on one channel that overlap in time (touching ones stay apart)."""
    user_code, users = pd.factorize(views["user_id"])
    df = pd.DataFrame({
        "user_code": user_code,
        "channel_id": views["channel_id"].to_numpy(),
        "start_s": to_seconds(views["start_time"]) + shift_s,
        "stop_s": to_seconds(views["stop_time"]) + shift_s,
    }).sort_values(["user_code", "channel_id", "start_s", "stop_s"], kind="stable", ignore_index=True)

    new_group = (df["user_code"].diff() != 0) | (df["channel_id"].diff() != 0)
    run_max = df.groupby(new_group.cumsum())["stop_s"].cummax()
    new_block = new_group | (df["start_s"] >= run_max.shift())
    merged = df.groupby(new_block.cumsum(), sort=False).agg(
        user_code=("user_code", "first"), channel_id=("channel_id", "first"),
        start_s=("start_s", "first"), stop_s=("stop_s", "max"), n_raw=("start_s", "size"),
    ).reset_index(drop=True)
    merged.insert(0, "user_id", users[merged["user_code"].to_numpy()])
    return merged


def overlap_pairs(sessions: pd.DataFrame, airings: pd.DataFrame) -> pd.DataFrame:
    """All (session, airing) pairs on the same channel with positive overlap, via searchsorted."""
    a_ch = airings["channel_id"].to_numpy().astype(np.int64) * CHANNEL_KEY
    a_start, a_end = airings["start_s"].to_numpy(), airings["end_s"].to_numpy()
    start_key = a_ch + a_start
    assert (np.diff(start_key) >= 0).all(), "airings must be sorted by (channel, start)"
    # running max of end keeps the array sorted even when airings on a channel overlap
    end_key_cummax = np.maximum.accumulate(a_ch + a_end)

    s_ch = sessions["channel_id"].to_numpy().astype(np.int64) * CHANNEL_KEY
    s_start, s_stop = sessions["start_s"].to_numpy(), sessions["stop_s"].to_numpy()
    lo = np.searchsorted(end_key_cummax, s_ch + s_start, side="right")  # before lo: every airing ends <= start
    hi = np.searchsorted(start_key, s_ch + s_stop, side="left")  # from hi: every airing starts >= stop
    n = np.clip(hi - lo, 0, None)

    sess_idx = np.repeat(np.arange(len(sessions)), n)
    air_idx = np.repeat(lo, n) + np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n)
    overlap = (np.minimum(s_stop[sess_idx], a_end[air_idx])
               - np.maximum(s_start[sess_idx], a_start[air_idx]))
    keep = overlap > 0
    return pd.DataFrame({"session_idx": sess_idx[keep], "airing_id": air_idx[keep], "overlap_s": overlap[keep]})


def airing_views(sessions: pd.DataFrame, airings: pd.DataFrame,
                 threshold: float = config.WATCH_THRESHOLD) -> pd.DataFrame:
    pairs = overlap_pairs(sessions, airings)
    pairs["user_code"] = sessions["user_code"].to_numpy()[pairs["session_idx"].to_numpy()]
    agg = pairs.groupby(["user_code", "airing_id"], sort=True).agg(
        overlap_s=("overlap_s", "sum"), n_sessions=("overlap_s", "size")).reset_index()

    users = sessions.drop_duplicates("user_code").set_index("user_code")["user_id"]
    air = airings.iloc[agg["airing_id"].to_numpy()].reset_index(drop=True)
    out = pd.DataFrame({
        "user_id": users.loc[agg["user_code"].to_numpy()].to_numpy(),
        "airing_id": agg["airing_id"].astype("int32").to_numpy(),
        "channel_id": air["channel_id"].to_numpy(),
        "tv_show_id": air["tv_show_id"].to_numpy(),
        "airing_start": air["start_time"].to_numpy(),
        "duration": air["duration"].to_numpy(),
        "overlap_s": np.minimum(agg["overlap_s"].to_numpy(), air["duration"].to_numpy()),
        "n_sessions": agg["n_sessions"].astype("int32").to_numpy(),
    })
    out["screen_time"] = out["overlap_s"] / out["duration"]
    out["watched"] = out["screen_time"] >= threshold
    out["week"] = out["airing_start"].dt.isocalendar()["week"].astype("int16")
    return out


def mode_per_item(labelled: pd.DataFrame, col: str) -> pd.Series:
    """Most frequent non-null value per tv_show_id; ties broken alphabetically."""
    counts = labelled.groupby(["tv_show_id", col]).size().rename("n").reset_index()
    counts = counts.sort_values(["tv_show_id", "n", col], ascending=[True, False, True])
    return counts.drop_duplicates("tv_show_id").set_index("tv_show_id")[col]


def build_items(schedule: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    labelled = schedule[schedule["tv_show_id"] != "0"]
    week = labelled["start_time"].dt.isocalendar()["week"]
    items = pd.DataFrame(index=pd.Index(sorted(labelled["tv_show_id"].unique()), name="tv_show_id"))
    for col in ITEM_ATTRS:
        items[col] = mode_per_item(labelled, col)
    for name, (lo, hi) in ITEM_WINDOWS.items():
        items[f"n_airings_{name}"] = (labelled[week.between(lo, hi)].groupby("tv_show_id").size()
                                      .reindex(items.index, fill_value=0).astype("int32"))
    n_values = labelled.groupby("tv_show_id")[ITEM_ATTRS].nunique()
    consistency = {f"items_with_several_{c}": int((n_values[c] > 1).sum()) for c in ITEM_ATTRS}
    return items.reset_index(), consistency


def histogram(screen_time: pd.Series) -> dict:
    counts, _ = np.histogram(screen_time, bins=HIST_BINS)  # last bin [0.9, 1.0] is closed
    hist = {f"[{a:.1f},{b:.1f}{']' if b == 1 else ')'}": int(c) for a, b, c in zip(HIST_BINS[:-1], HIST_BINS[1:], counts)}
    hist["exactly_1.0"] = int((screen_time == 1).sum())
    return hist


def no_watched(users: set, views: pd.DataFrame, max_week: int) -> int:
    watched_users = set(views.loc[views["watched"] & (views["week"] <= max_week), "user_id"])
    return len(users - watched_users)


def main():
    t0, timing = time.time(), {}
    I = config.INTERIM_DIR
    views = pd.read_parquet(I / "log.parquet")
    schedules = {p: canonical_ids(pd.read_parquet(I / f"schedule_{p}.parquet")) for p in config.SCHEDULE_FILES}

    train_sched, n_dup_train = dedup_schedule([schedules["11-20"], schedules["21-30"]])
    airings = build_airings(train_sched)
    sessions = merge_sessions(views, shift_s=config.TIME_SHIFT_HOURS * 3600)
    timing["prepare_s"] = round(time.time() - t0, 1)
    log.info("sessions %d -> %d after merge; airings %d", len(views), len(sessions), len(airings))

    t1 = time.time()
    av = airing_views(sessions, airings)
    timing["join_s"] = round(time.time() - t1, 1)
    log.info("pairs %d, watched %d", len(av), int(av["watched"].sum()))

    all_sched, n_dup_all = dedup_schedule(list(schedules.values()))
    items, consistency = build_items(all_sched)

    I.mkdir(parents=True, exist_ok=True)
    av.to_parquet(I / "airing_views.parquet", index=False)
    items.to_parquet(I / "items.parquet", index=False)

    sub_users = set(pd.read_csv(config.SUBMISSION_TEMPLATE, encoding=config.ENCODING, dtype=str)["user_id"])
    w = av[av["watched"]]
    stats = {
        "rule": {"watch_threshold": config.WATCH_THRESHOLD, "time_shift_hours": config.TIME_SHIFT_HOURS,
                 "week": "ISO week of airing start", "merge": "sessions of one user on one channel merged when start < running max stop"},
        "sessions": {"raw": len(views), "exact_duplicate_rows_raw": int(views.duplicated().sum()),
                     "after_merge": len(sessions), "merged_away": len(views) - len(sessions),
                     "merged_sessions_with_2plus_raw": int((sessions["n_raw"] > 1).sum())},
        "airings": {"train_schedule_rows": sum(len(schedules[p]) for p in ["11-20", "21-30"]),
                    "duplicate_channel_start_removed": n_dup_train,
                    "labelled_airings": len(airings),
                    "labelled_overlapping_next_on_channel": int(
                        ((airings["channel_id"].diff(-1) == 0) & (airings["start_s"].shift(-1) < airings["end_s"])).sum())},
        "pairs": {"total": len(av), "watched": len(w),
                  "users": int(av["user_id"].nunique()), "users_with_watched": int(w["user_id"].nunique()),
                  "programmes": int(av["tv_show_id"].nunique()), "programmes_watched": int(w["tv_show_id"].nunique()),
                  "pairs_with_2plus_sessions": int((av["n_sessions"] > 1).sum()),
                  "by_week_watched": {int(k): int(v) for k, v in w["week"].value_counts().sort_index().items()}},
        "screen_time_histogram": histogram(av["screen_time"]),
        "watched_pairs_by_threshold": {str(t): int((av["screen_time"] >= t).sum()) for t in THRESHOLDS},
        "submission_cold_start": {
            "submission_users": len(sub_users),
            "no_watched_weeks_le_20": no_watched(sub_users, av, 20),
            "no_watched_weeks_le_30": no_watched(sub_users, av, 30),
            "no_pair_at_all_weeks_le_30": len(sub_users - set(av["user_id"])),
            "pairs_from_week_10_airings": int((av["week"] == 10).sum()),
        },
        "items": {"rows": len(items), "schedule_duplicate_channel_start_removed": n_dup_all, **consistency},
        "runtime": {**timing, "total_s": round(time.time() - t0, 1)},
    }
    out_path = config.OUTPUTS_DIR / "join_stats.json"
    out_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding=config.ENCODING)
    log.info("saved %s in %.1f s", out_path, stats["runtime"]["total_s"])


if __name__ == "__main__":
    main()
