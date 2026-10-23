"""Manual check of random (user, airing) pairs against the raw CSV files, in plain Python.

Independent of src/join.py: raw rows are read with the csv module, times parsed with datetime,
intervals clipped to the airing and unioned by hand. Run: python -m src.check_pair
"""
import csv
from datetime import datetime, timedelta

import pandas as pd

from src import config

LOG_FMT = "%Y-%m-%d %H:%M:%S"
SCHED_FMT = "%d.%m.%Y %H:%M:%S"

log = config.get_logger("03_check_pair")


def raw_log_rows(user_id: str, channel_id: int) -> list[list[str]]:
    with open(config.VIEWS_FILE, encoding=config.ENCODING, newline="") as f:
        reader = csv.reader(f)
        next(reader)
        return [r for r in reader if r[0] == user_id and int(r[1]) == channel_id]


def raw_schedule_rows(channel_id: int, start: datetime) -> list[tuple[str, list[str]]]:
    key = start.strftime(SCHED_FMT)
    rows = []
    for period in ["11-20", "21-30"]:
        with open(config.SCHEDULE_FILES[period], encoding=config.ENCODING, newline="") as f:
            reader = csv.reader(f)
            next(reader)
            rows += [(period, r) for r in reader if int(r[0]) == channel_id and r[2] == key]
    return rows


def check(pair: pd.Series) -> bool:
    user, ch = pair["user_id"], int(pair["channel_id"])
    log.info("=" * 70)
    log.info("pair: user %s, channel %d, tv_show_id %s, airing_start %s", user, ch, pair["tv_show_id"], pair["airing_start"])

    sched = raw_schedule_rows(ch, pair["airing_start"].to_pydatetime())
    for period, r in sched:
        log.info("raw schedule [%s]: %s", period, r[:7])
    labelled = [r for _, r in sched if r[5] != "0"]
    assert len(labelled) == 1, f"expected one labelled schedule row, got {len(labelled)}"
    a_start = datetime.strptime(labelled[0][2], SCHED_FMT)
    duration = int(labelled[0][3])
    a_end = a_start + timedelta(seconds=duration)
    log.info("step 1  airing [%s, %s), duration %d s", a_start, a_end, duration)

    rows = raw_log_rows(user, ch)
    log.info("step 2  raw log rows of this user on channel %d: %d", ch, len(rows))
    clipped = []
    for r in rows:
        s, e = datetime.strptime(r[2], LOG_FMT), datetime.strptime(r[3], LOG_FMT)
        lo, hi = max(s, a_start), min(e, a_end)
        if hi > lo:
            log.info("        raw %s  ->  clipped to airing [%s, %s) = %d s", r, lo.time(), hi.time(), (hi - lo).total_seconds())
            clipped.append((lo, hi))

    clipped.sort()
    union = []
    for lo, hi in clipped:  # union of clipped intervals; strictly overlapping ones are merged
        if union and lo < union[-1][1]:
            union[-1][1] = max(union[-1][1], hi)
        else:
            union.append([lo, hi])
    overlap = sum((hi - lo).total_seconds() for lo, hi in union)
    log.info("step 3  union of clipped intervals: %s", [(lo.time().isoformat(), hi.time().isoformat()) for lo, hi in union])
    overlap = min(overlap, duration)
    screen_time = overlap / duration
    watched = screen_time >= config.WATCH_THRESHOLD
    log.info("step 4  overlap = %d s, screen_time = %d / %d = %.4f, watched = %s", overlap, overlap, duration, screen_time, watched)

    ok = (overlap == pair["overlap_s"] and abs(screen_time - pair["screen_time"]) < 1e-12
          and watched == pair["watched"] and len(union) == pair["n_sessions"])
    log.info("parquet: overlap_s %d, screen_time %.4f, watched %s, n_sessions %d -> %s",
             pair["overlap_s"], pair["screen_time"], pair["watched"], pair["n_sessions"], "MATCH" if ok else "MISMATCH")
    return ok


def main():
    av = pd.read_parquet(config.INTERIM_DIR / "airing_views.parquet")
    picks = [av.sample(1, random_state=config.SEED).iloc[0],
             av[av["n_sessions"] > 1].sample(1, random_state=config.SEED).iloc[0]]
    results = [check(p) for p in picks]
    log.info("all match: %s", all(results))


if __name__ == "__main__":
    main()
