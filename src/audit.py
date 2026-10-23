"""Data audit: row counts, dates, label coverage, date-parsing trap, timezone check, parquet cache.

Run from the project root:  python -m src.audit
Writes outputs/data_audit.json, outputs/interim/log.parquet, outputs/interim/schedule_<period>.parquet
"""
import json
import time
import warnings

import numpy as np
import pandas as pd
import psutil

from src import config

LOG_DT_FORMAT = "%Y-%m-%d %H:%M:%S"
SCHEDULE_DT_FORMAT = "%d.%m.%Y %H:%M:%S"
QUANTILES = [0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1]
INT64_MAX = 2**63 - 1

DATE_TRAP_SAMPLE = 10_000
TZ_SAMPLE = 200_000
TZ_TOLERANCE_S = 120
TZ_MIN_MARGIN_PP = 2.0  # "clearly better" = ahead of the runner-up by >= 2 p.p. for starts AND ends
DST_SWITCH = pd.Timestamp("2020-03-29")  # Ukraine switched to summer time on 29.03.2020
NOMINAL_WEEKS = {"11-20": (11, 20), "21-30": (21, 30), "31-42": (31, 42)}

SCHEDULE_STR_COLS = [
    "channel_title", "start_time", "tv_show_title", "tv_show_id", "tv_show_category",
    "tv_show_genre_1", "tv_show_genre_2", "tv_show_genre_3", "year_of_production",
    "director", "actors",
]

# Figures stated in exam_quiz_GSB.md, used only for cross-checking
TASK_FIGURES = {
    "rows": {"log": 5_821_987, "11-20": 1_074_235, "21-30": 1_074_394, "31-42": 1_295_378},
    "submission_users": 1_260,
    "labelled_rows_11-20": 115_613,
    "distinct_programmes": {"11-20": 5_848, "21-30": 6_364, "31-42": 6_637},
    "channels": {"11-20": 400, "21-30": 403, "31-42": 404},
    "duraton_median": 396,
    "duraton_max": 342_145,
}

log = config.get_logger("01_audit")


def to_seconds(s: pd.Series) -> np.ndarray:
    return s.dt.as_unit("s").astype("int64").to_numpy()


def quantiles(s: pd.Series) -> dict:
    return {f"{q:g}": float(v) for q, v in s.quantile(QUANTILES).items()}


def read_log() -> pd.DataFrame:
    df = pd.read_csv(config.VIEWS_FILE, encoding=config.ENCODING,
                     dtype={"user_id": str, "start_time": str, "stop_time": str})
    assert df.isna().sum().sum() == 0, "nulls in view log"
    assert df["vsetv_id"].between(np.iinfo("int32").min, np.iinfo("int32").max).all()
    df["vsetv_id"] = df["vsetv_id"].astype("int32")
    df["start_time"] = pd.to_datetime(df["start_time"], format=LOG_DT_FORMAT)
    df["stop_time"] = pd.to_datetime(df["stop_time"], format=LOG_DT_FORMAT)
    return df


def read_schedule(path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding=config.ENCODING, dtype={c: str for c in SCHEDULE_STR_COLS})
    assert df["channel_id"].notna().all() and df["start_time"].notna().all()
    df["channel_id"] = df["channel_id"].astype("int32")
    df["start_time"] = pd.to_datetime(df["start_time"], format=SCHEDULE_DT_FORMAT)
    return df


def date_range(s: pd.Series) -> dict:
    iso = s.dt.isocalendar()
    return {
        "min": str(s.min()), "max": str(s.max()),
        "iso_years": sorted(int(y) for y in iso["year"].unique()),
        "iso_week_min": int(iso["week"].min()), "iso_week_max": int(iso["week"].max()),
        "n_iso_weeks": int(iso["week"].nunique()),
    }


def audit_log(df: pd.DataFrame) -> dict:
    uid_int = df["user_id"].map(int)
    real_dur = (df["stop_time"] - df["start_time"]).dt.total_seconds()
    return {
        "rows": len(df),
        "users": int(df["user_id"].nunique()),
        "channels": int(df["vsetv_id"].nunique()),
        "user_id_digits_max": int(df["user_id"].str.len().max()),
        "user_id_above_int64_max_rows": int((uid_int > INT64_MAX).sum()),
        "start_time": date_range(df["start_time"]),
        "stop_time": {"min": str(df["stop_time"].min()), "max": str(df["stop_time"].max())},
        "duraton_quantiles_s": quantiles(df["duraton"]),
        "duraton_ne_stop_minus_start_rows": int((real_dur != df["duraton"]).sum()),
        "duraton_mismatch_diff_s": {f"{k:g}": int(n) for k, n in
                                    (real_dur - df["duraton"])[real_dur != df["duraton"]].value_counts().items()},
        "duraton_mismatch_start_dates": {str(k): int(n) for k, n in
                                         df.loc[real_dur != df["duraton"], "start_time"].dt.date.value_counts().items()},
        "duraton_le_0_rows": int((df["duraton"] <= 0).sum()),
        "sessions_crossing_midnight": int((df["stop_time"].dt.normalize() != df["start_time"].dt.normalize()).sum()),
        "exact_duplicate_rows": int(df.duplicated().sum()),
    }


def audit_schedule(df: pd.DataFrame, period: str) -> dict:
    labelled = df["tv_show_id"] != "0"
    week = df["start_time"].dt.isocalendar()["week"]
    outside = ~week.between(*NOMINAL_WEEKS[period])
    end = df["start_time"] + pd.to_timedelta(df["duration"], unit="s")
    srt = df.assign(end=end).sort_values(["channel_id", "start_time"])
    same_ch = srt["channel_id"].eq(srt["channel_id"].shift(-1))
    nxt = srt["start_time"].shift(-1)
    return {
        "rows": len(df),
        "channels": int(df["channel_id"].nunique()),
        "start_time": date_range(df["start_time"]),
        "tv_show_id_non_numeric_rows": int((~df["tv_show_id"].str.fullmatch(r"\d+")).sum()),
        "labelled_rows": int(labelled.sum()),
        "unlabelled_rows": int((~labelled).sum()),
        "labelled_share": round(float(labelled.mean()), 4),
        "distinct_programmes": int(df.loc[labelled, "tv_show_id"].nunique()),
        "distinct_tv_show_id_incl_0": int(df["tv_show_id"].nunique()),
        "rows_outside_nominal_weeks": int(outside.sum()),
        "labelled_rows_outside_nominal_weeks": int((outside & labelled).sum()),
        "outside_rows_by_week": {int(k): int(n) for k, n in week[outside].value_counts().sort_index().items()},
        "categories": sorted(df["tv_show_category"].dropna().unique().tolist()),
        "duration_quantiles_s": quantiles(df["duration"]),
        "duration_le_0_rows": int((df["duration"] <= 0).sum()),
        "duration_null_rows": int(df["duration"].isna().sum()),
        "exact_duplicate_rows": int(df.duplicated().sum()),
        "duplicate_channel_start_rows": int(df.duplicated(["channel_id", "start_time"]).sum()),
        "overlap_with_next_rows": int((same_ch & (nxt < srt["end"])).sum()),
        "gap_before_next_rows": int((same_ch & (nxt > srt["end"])).sum()),
    }


def try_vectorised(values: pd.Series) -> dict:
    """pd.to_datetime without format: pandas infers ONE format from the first value."""
    correct = pd.to_datetime(values, format=SCHEDULE_DT_FORMAT)
    out = {"first_value": values.iloc[0], "n": len(values)}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            parsed = pd.to_datetime(values)
            out.update(error=None, mismatches=int((parsed != correct).sum()))
        except Exception as e:  # noqa: BLE001 — the failure itself is the finding
            out.update(error=f"{type(e).__name__}: {str(e).splitlines()[0]}", mismatches=None)
    out["warnings"] = [str(w.message) for w in caught]
    return out


def audit_date_trap(raw: pd.Series) -> dict:
    """Parse the same schedule dates with and without an explicit format."""
    sample = raw.sample(DATE_TRAP_SAMPLE, random_state=config.SEED)
    correct = pd.to_datetime(sample, format=SCHEDULE_DT_FORMAT)
    ambiguous = (correct.dt.day <= 12) & (correct.dt.day != correct.dt.month)
    first_amb = sample[ambiguous].index[0]
    out = {"sample_size": DATE_TRAP_SAMPLE, "source": "export_arh_11-20-final.csv"}

    out["vectorised_no_format"] = {
        "seed_sample_as_is": try_vectorised(sample),
        "seed_sample_ambiguous_first": try_vectorised(
            pd.concat([sample.loc[[first_amb]], sample.drop(first_amb)])),
        "full_file_original_order": try_vectorised(raw),
    }

    elem = sample.map(pd.Timestamp)  # element-wise: dateutil guesses each value separately
    mism = elem != correct
    out["elementwise_no_format"] = {
        "mismatches": int(mism.sum()),
        "mismatch_share": round(float(mism.mean()), 4),
        "ambiguous_in_sample": int(ambiguous.sum()),
        "examples": [{"raw": r, "correct": str(c), "parsed": str(p)}
                     for r, c, p in zip(sample[mism][:3], correct[mism][:3], elem[mism][:3])],
    }
    return out


def channel_boundaries(sched: pd.DataFrame) -> dict:
    start = to_seconds(sched["start_time"])
    bounds = pd.DataFrame({
        "channel_id": np.concatenate([sched["channel_id"].to_numpy()] * 2),
        "t": np.concatenate([start, start + sched["duration"].to_numpy()]),
    }).drop_duplicates()
    return {ch: np.sort(g["t"].to_numpy()) for ch, g in bounds.groupby("channel_id")}


def near_boundary(channels: np.ndarray, t: np.ndarray, bounds: dict) -> np.ndarray:
    """True where t is within TZ_TOLERANCE_S of a programme boundary on the same channel."""
    hit = np.zeros(len(t), dtype=bool)
    for ch in np.unique(channels):
        idx = np.flatnonzero(channels == ch)
        b = bounds[ch]
        pos = np.clip(np.searchsorted(b, t[idx]), 1, len(b) - 1)
        dist = np.minimum(np.abs(t[idx] - b[pos - 1]), np.abs(t[idx] - b[pos]))
        hit[idx] = dist <= TZ_TOLERANCE_S
    return hit


def audit_timezone(views: pd.DataFrame, sched: pd.DataFrame) -> dict:
    bounds = channel_boundaries(sched)
    known = views["vsetv_id"].isin(list(bounds))
    sample = views[known].sample(TZ_SAMPLE, random_state=config.SEED)
    ch = sample["vsetv_id"].to_numpy()
    start, stop = to_seconds(sample["start_time"]), to_seconds(sample["stop_time"])
    before_dst = (sample["start_time"] < DST_SWITCH).to_numpy()

    variants = {f"{h:+d}h": np.full(len(sample), h * 3600) for h in range(-3, 4)}
    variants["dst_-2h/-3h"] = np.where(before_dst, -2 * 3600, -3 * 3600)
    variants["dst_+2h/+3h"] = np.where(before_dst, 2 * 3600, 3 * 3600)  # mirror, added for symmetry

    table = {}
    for name, shift in variants.items():
        table[name] = {
            "starts_within_120s_pct": round(100 * near_boundary(ch, start + shift, bounds).mean(), 2),
            "ends_within_120s_pct": round(100 * near_boundary(ch, stop + shift, bounds).mean(), 2),
        }
        log.info("tz %-12s starts %5.2f%%  ends %5.2f%%", name, *table[name].values())

    def leader(key):
        ranked = sorted(table, key=lambda v: table[v][key], reverse=True)
        return ranked[0], table[ranked[0]][key] - table[ranked[1]][key]

    (best_s, margin_s), (best_e, margin_e) = leader("starts_within_120s_pct"), leader("ends_within_120s_pct")
    clear = best_s == best_e and margin_s >= TZ_MIN_MARGIN_PP and margin_e >= TZ_MIN_MARGIN_PP
    chosen = best_s if clear else "+0h"
    return {
        "sample_size": TZ_SAMPLE, "seed": config.SEED, "tolerance_s": TZ_TOLERANCE_S,
        "sessions_on_channels_without_schedule": int((~known).sum()),
        "schedule_used": "11-20 + 21-30, all rows (boundaries do not depend on tv_show_id)",
        "rule": f"apply a shift only if the same variant leads for starts and ends by >= {TZ_MIN_MARGIN_PP} p.p.",
        "table": table,
        "best_for_starts": best_s, "margin_starts_pp": round(margin_s, 2),
        "best_for_ends": best_e, "margin_ends_pp": round(margin_e, 2),
        "clear_winner": bool(clear),
        "decision": chosen,
    }


def audit_submission(views: pd.DataFrame) -> dict:
    sub = pd.read_csv(config.SUBMISSION_TEMPLATE, encoding=config.ENCODING, dtype=str)
    users = set(sub["user_id"])
    week = views["start_time"].dt.isocalendar()["week"]
    u_1120 = set(views.loc[week.between(11, 20), "user_id"])
    u_2130 = set(views.loc[week.between(21, 30), "user_id"])
    return {
        "rows": len(sub),
        "columns": sub.columns.tolist(),
        "distinct_users": len(users),
        "matches_task_1260": len(users) == TASK_FIGURES["submission_users"],
        "in_log_weeks_11_20": len(users & u_1120),
        "in_log_weeks_21_30": len(users & u_2130),
        "in_log_any": len(users & (u_1120 | u_2130)),
        "in_both_periods": len(users & u_1120 & u_2130),
        "absent_from_log": len(users - u_1120 - u_2130),
        "note": "weeks from unshifted log start_time",
    }


def save_cache(views: pd.DataFrame, schedules: dict) -> dict:
    config.INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    out = {}
    cached = views.rename(columns={"vsetv_id": "channel_id"})
    cached["duraton"] = cached["duraton"].astype("int32")
    files = {"log.parquet": cached}
    files.update({f"schedule_{p}.parquet": df for p, df in schedules.items()})
    for name, df in files.items():
        path = config.INTERIM_DIR / name
        df.to_parquet(path, index=False)
        back = pd.read_parquet(path)
        assert len(back) == len(df) and back["channel_id"].dtype == "int32"
        out[name] = {"rows": len(back), "mb": round(path.stat().st_size / 2**20, 1),
                     "dtypes": {c: str(t) for c, t in back.dtypes.items()}}
        log.info("cached %s: %d rows, %.1f MB", name, len(back), out[name]["mb"])
    return out


def main():
    t0 = time.time()
    views = read_log()
    log.info("log read: %d rows", len(views))
    schedules = {p: read_schedule(path) for p, path in config.SCHEDULE_FILES.items()}

    audit = {"log": audit_log(views)}
    audit["schedule"] = {p: audit_schedule(df, p) for p, df in schedules.items()}

    train_sched = pd.concat([schedules["11-20"], schedules["21-30"]], ignore_index=True)
    train_check = audit_schedule(train_sched, "11-20")
    audit["schedule_11-20_vs_21-30_boundary"] = {
        "exact_duplicate_rows": train_check["exact_duplicate_rows"],
        "duplicate_channel_start_rows": train_check["duplicate_channel_start_rows"],
        "overlap_with_next_rows_after_concat": train_check["overlap_with_next_rows"],
        "max_start_11-20": str(schedules["11-20"]["start_time"].max()),
        "min_start_21-30": str(schedules["21-30"]["start_time"].min()),
    }
    sched_channels = set(train_sched["channel_id"])
    audit["log_channels_in_train_schedule"] = {
        "log_channels": int(views["vsetv_id"].nunique()),
        "found_in_schedule": int(views["vsetv_id"].drop_duplicates().isin(sched_channels).sum()),
        "log_rows_on_found_channels_share": round(float(views["vsetv_id"].isin(sched_channels).mean()), 4),
    }
    audit["submission"] = audit_submission(views)

    raw_1120 = pd.read_csv(config.SCHEDULE_FILES["11-20"], encoding=config.ENCODING,
                           usecols=["start_time"], dtype=str)["start_time"]
    audit["date_parsing_trap"] = audit_date_trap(raw_1120)
    audit["timezone"] = audit_timezone(views, train_sched)

    a = audit
    audit["check_vs_task"] = {
        "rows": {k: [TASK_FIGURES["rows"][k], (a["log"] if k == "log" else a["schedule"][k])["rows"]]
                 for k in TASK_FIGURES["rows"]},
        "labelled_rows_11-20": [TASK_FIGURES["labelled_rows_11-20"], a["schedule"]["11-20"]["labelled_rows"]],
        "distinct_programmes": {p: [v, a["schedule"][p]["distinct_programmes"]]
                                for p, v in TASK_FIGURES["distinct_programmes"].items()},
        "distinct_tv_show_id_incl_0": {p: [v, a["schedule"][p]["distinct_tv_show_id_incl_0"]]
                                       for p, v in TASK_FIGURES["distinct_programmes"].items()},
        "channels": {p: [v, a["schedule"][p]["channels"]] for p, v in TASK_FIGURES["channels"].items()},
        "duraton_median": [TASK_FIGURES["duraton_median"], a["log"]["duraton_quantiles_s"]["0.5"]],
        "duraton_max": [TASK_FIGURES["duraton_max"], a["log"]["duraton_quantiles_s"]["1"]],
        "_format": "[task, observed]",
    }

    audit["cache"] = save_cache(views, schedules)
    audit["runtime"] = {"seconds": round(time.time() - t0, 1),
                        "rss_at_end_mb": round(psutil.Process().memory_info().rss / 2**20)}
    out_path = config.OUTPUTS_DIR / "data_audit.json"
    out_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding=config.ENCODING)
    log.info("saved %s in %.1f s", out_path, audit["runtime"]["seconds"])


if __name__ == "__main__":
    main()
