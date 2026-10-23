import math

import numpy as np
import pandas as pd

from src.slots import fuse, slot_key, slot_profile, slot_scores

MON_20 = pd.Timestamp("2020-03-09 20:15")  # Monday, 20:xx
TUE_21 = pd.Timestamp("2020-03-10 21:00")  # Tuesday, 21:xx


def test_slot_key():
    # channel 3, Monday (0) 20h -> 3 * 168 + 0 * 24 + 20 = 524; Tuesday 21h -> 3 * 168 + 24 + 21 = 549
    keys = slot_key(pd.Series([3, 3]), pd.Series([MON_20, TUE_21]))
    assert keys.tolist() == [524, 549]


def test_profile_share_of_weighted_weeks_by_hand():
    # window weeks 19-20, H = 1: w19 = 0.5, w20 = 1, denominator 1.5
    # u watched slot 524 in week 20 twice (signals 0.4 and 0.9 -> max 0.9) and in week 19 (signal 1.0)
    # p = (1 * 0.9 + 0.5 * 1.0) / 1.5 = 0.9333
    train = pd.DataFrame({"user_id": ["u", "u", "u"], "channel_id": [3, 3, 3],
                          "airing_start": [MON_20 + pd.Timedelta(weeks=11), MON_20 + pd.Timedelta(weeks=11, minutes=30),
                                           MON_20 + pd.Timedelta(weeks=10)],
                          "week": [20, 20, 19], "watched": [True, True, True], "signal": [0.4, 0.9, 1.0]})
    p = slot_profile(train, half_life=1.0, train_weeks=(19, 20))
    assert abs(p[("u", 524)] - (0.9 + 0.5) / 1.5) < 1e-12


def test_slot_scores_expected_count():
    # profile: u has 0.5 for slot 524 and 0.2 for 549
    # programme "7": 2 airings in 524 + 1 in 549 -> 2 * 0.5 + 0.2 = 1.2; programme "8": 1 airing in 549 -> 0.2
    # programme "9": airs only in a slot u never watches -> 0 -> not a candidate
    profile = pd.Series([0.5, 0.2], index=pd.MultiIndex.from_tuples([("u", 524), ("u", 549)], names=["user_id", "key"]),
                        name="num")
    airings = pd.DataFrame({"tv_show_id": ["7", "7", "8", "9"], "key": [524, 549, 549, 1], "n": [2, 1, 1, 3]})
    ranked, scores = slot_scores(profile, airings, ["u"])
    assert ranked == {"u": ["7", "8"]}
    np.testing.assert_allclose(scores["u"], [1.2, 0.2])


def test_fuse_lambda_inf_zero_and_mix():
    v5r, v5s = {"u": ["a", "b"]}, {"u": np.array([2.0, 1.0])}    # normalised 1.0, 0.5
    sr, ss = {"u": ["c", "b"]}, {"u": np.array([4.0, 1.0])}      # normalised 1.0, 0.25
    assert fuse(v5r, v5s, sr, ss, math.inf) == {"u": ["a", "b"]}           # v5 only
    assert fuse(v5r, v5s, sr, ss, 0.0) == {"u": ["c", "b"]}                # slot only (a scores 0)
    # lambda 1: a 0 + 1.0 = 1.0, b 0.25 + 0.5 = 0.75, c 1.0 -> a and c tie -> candidate order a, c; then b
    assert fuse(v5r, v5s, sr, ss, 1.0) == {"u": ["a", "c", "b"]}
