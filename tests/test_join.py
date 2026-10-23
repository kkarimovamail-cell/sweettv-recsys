"""Join rule on hand-made data. All times are on 2020-03-09; expected values are worked out in comments."""
import pandas as pd
import pytest

from src.join import airing_views, build_airings, build_items, canonical_ids, dedup_schedule, merge_sessions

DAY = "2020-03-09 "


def ts(hhmm: str) -> pd.Timestamp:
    return pd.Timestamp(DAY + hhmm)


def make_views(rows):  # rows: (user, channel, start, stop)
    return pd.DataFrame({
        "user_id": [r[0] for r in rows],
        "channel_id": pd.array([r[1] for r in rows], dtype="int32"),
        "start_time": [ts(r[2]) for r in rows],
        "stop_time": [ts(r[3]) for r in rows],
    })


def make_schedule(rows):  # rows: (channel, start, duration_s, tv_show_id)
    return pd.DataFrame({
        "channel_id": pd.array([r[0] for r in rows], dtype="int32"),
        "start_time": [ts(r[1]) for r in rows],
        "duration": [r[2] for r in rows],
        "tv_show_id": [r[3] for r in rows],
    })


def run(view_rows, schedule_rows):
    out = airing_views(merge_sessions(make_views(view_rows)), build_airings(make_schedule(schedule_rows)))
    return {(r.user_id, r.tv_show_id): (r.overlap_s, round(r.screen_time, 4), r.watched) for r in out.itertuples()}


def test_session_covers_whole_airing():
    # airing 10:00-11:00 (3600 s), session 09:30-11:30 -> overlap 3600, screen_time 1.0
    assert run([("u", 1, "09:30", "11:30")], [(1, "10:00", 3600, "A")]) == {("u", "A"): (3600, 1.0, True)}


def test_airing_starts_mid_session():
    # session 09:50-10:50; A 09:00-10:00 -> 600/3600 = 0.1667; B 10:00-11:00 -> 3000/3600 = 0.8333
    got = run([("u", 1, "09:50", "10:50")], [(1, "09:00", 3600, "A"), (1, "10:00", 3600, "B")])
    assert got == {("u", "A"): (600, 0.1667, False), ("u", "B"): (3000, 0.8333, True)}


def test_overlapping_sessions_are_merged_not_double_counted():
    # 10:00-10:40 and 10:20-10:50 merge into 10:00-10:50 -> 3000/3600 = 0.8333
    # (without merging: 2400 + 1800 = 4200 -> clipped 1.0, which would be wrong)
    got = run([("u", 1, "10:00", "10:40"), ("u", 1, "10:20", "10:50")], [(1, "10:00", 3600, "A")])
    assert got == {("u", "A"): (3000, 0.8333, True)}


def test_disjoint_sessions_are_summed():
    # 10:00-10:30 (1800) + 10:35-11:00 (1500) = 3300/3600 = 0.9167
    got = run([("u", 1, "10:00", "10:30"), ("u", 1, "10:35", "11:00")], [(1, "10:00", 3600, "A")])
    assert got == {("u", "A"): (3300, 0.9167, True)}


def test_threshold_is_inclusive_and_touching_gives_no_pair():
    # u: 10:00-10:48 -> 2880/3600 = 0.8 exactly -> watched (>= 0.8)
    # v: 11:00-11:30 only touches the end of A (11:00) -> overlap 0 -> no pair
    got = run([("u", 1, "10:00", "10:48"), ("v", 1, "11:00", "11:30")], [(1, "10:00", 3600, "A")])
    assert got == {("u", "A"): (2880, 0.8, True)}


def test_other_channel_unlabelled_and_other_user_do_not_mix():
    # channel 2 has only unlabelled airtime (tv_show_id "0") -> no pair for u on channel 2
    # u 10:00-10:30 and w 10:20-10:50 on channel 1 are different users -> not merged: 0.5 and 0.5
    got = run(
        [("u", 2, "10:00", "11:00"), ("u", 1, "10:00", "10:30"), ("w", 1, "10:20", "10:50")],
        [(1, "10:00", 3600, "A"), (2, "10:00", 3600, "0")],
    )
    assert got == {("u", "A"): (1800, 0.5, False), ("w", "A"): (1800, 0.5, False)}


def test_overlapping_airings_on_one_channel():
    # X 10:00-12:00 (7200), Y 11:00-11:30 (inside X), Z 12:00-13:00; session 11:50-12:10
    # X: 11:50-12:00 = 600 -> 0.0833; Y ends 11:30 -> no pair; Z: 12:00-12:10 = 600 -> 0.1667
    got = run([("u", 1, "11:50", "12:10")], [(1, "10:00", 7200, "X"), (1, "11:00", 1800, "Y"), (1, "12:00", 3600, "Z")])
    assert got == {("u", "X"): (600, 0.0833, False), ("u", "Z"): (600, 0.1667, False)}


def test_seam_duplicate_removed_and_item_mode():
    # same (channel, start) in two files -> one row kept; title mode: "T" twice vs "t" once -> "T"
    a = make_schedule([(1, "10:00", 3600, "A"), (1, "12:00", 3600, "A")]).assign(tv_show_title=["T", "T"])
    b = make_schedule([(1, "10:00", 3600, "A"), (1, "14:00", 3600, "A")]).assign(tv_show_title=["T", "t"])
    sched, n_dup = dedup_schedule([a, b])
    assert n_dup == 1 and len(sched) == 3
    for col in ["tv_show_category", "tv_show_genre_1", "tv_show_genre_2", "tv_show_genre_3", "director", "actors"]:
        sched[col] = None
    items, _ = build_items(sched)
    assert items.loc[0, "tv_show_title"] == "T" and items.loc[0, "n_airings_w11_20"] == 3


@pytest.mark.parametrize("shift_h", [1])
def test_time_shift_moves_sessions(shift_h):
    # session 09:00-10:00 shifted +1 h becomes 10:00-11:00 -> full overlap with A
    views = make_views([("u", 1, "09:00", "10:00")])
    out = airing_views(merge_sessions(views, shift_s=shift_h * 3600), build_airings(make_schedule([(1, "10:00", 3600, "A")])))
    assert out["screen_time"].tolist() == [1.0]


def test_canonical_ids_strip_leading_zeros_and_reject_collisions():
    # "00078499" -> "78499", "0" stays "0" (unlabelled); "007" and "7" would both become "7" -> error
    sched = make_schedule([(1, "10:00", 60, "00078499"), (1, "11:00", 60, "0")])
    assert canonical_ids(sched)["tv_show_id"].tolist() == ["78499", "0"]
    with pytest.raises(AssertionError):
        canonical_ids(make_schedule([(1, "10:00", 60, "007"), (1, "11:00", 60, "7")]))
