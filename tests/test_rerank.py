from types import SimpleNamespace

import numpy as np
import pandas as pd

from src.rerank import FEATURES, add_labels, lists_to_frame, n_weeks, rank_by
from src.rerank_v8 import predict_score, within_user_rank


def test_window_lengths():
    # train 11-20 -> 10 weeks, 11-30 -> 20; target 21-30 -> 10, 31-42 -> 12
    assert [n_weeks(w) for w in [(11, 20), (11, 30), (21, 30), (31, 42)]] == [10, 20, 10, 12]


def test_labels_5_to_1_by_truth_place():
    df = pd.DataFrame({"user_id": ["u"] * 3, "tv_show_id": ["a", "c", "z"], "cand_pos": [0, 1, 2]})
    truth = {"u": ["c", "b", "a", "d", "e"]}  # c -> 5, a -> 3, z not in truth -> 0
    assert add_labels(df, truth)["label"].tolist() == [3, 5, 0]


def test_lists_to_frame_normalises_by_first_and_ranks():
    df = lists_to_frame({"u": ["x", "y"]}, {"u": np.array([4.0, 1.0])}, "s", "r")
    assert df.to_dict("records") == [{"user_id": "u", "tv_show_id": "x", "s": 1.0, "r": 1},
                                     {"user_id": "u", "tv_show_id": "y", "s": 0.25, "r": 2}]


def test_rank_by_score_then_candidate_order_and_fill():
    # scores: a 0.5, b 0.9, c 0.5 -> b, then a (cand_pos 0) before c (cand_pos 2); top-up from popularity
    df = pd.DataFrame({"user_id": ["u"] * 3, "tv_show_id": ["a", "b", "c"], "cand_pos": [0, 1, 2]})
    run = SimpleNamespace(popular=["p", "q", "a", "b", "c"], pool={"a", "b", "c", "p", "q"})
    assert rank_by(df, np.array([0.5, 0.9, 0.5]), ["u", "cold"], run) == {
        "u": ["b", "a", "c", "p", "q"], "cold": ["p", "q", "a", "b", "c"]}


def test_mean_rank_averaging_by_hand():
    # one user, 3 candidates; model 1 scores (0.9, 0.5, 0.1) -> ranks (1, 2, 3); model 2 (0.1, 0.8, 0.9) -> (3, 2, 1)
    # mean ranks (2, 2, 2) -> all tie -> candidate order; score = -mean rank
    df = pd.DataFrame({"user_id": ["u"] * 3, "cand_pos": [0, 1, 2]})
    assert within_user_rank(df, np.array([0.9, 0.5, 0.1])).tolist() == [1, 2, 3]
    m1 = SimpleNamespace(predict=lambda X: np.array([0.9, 0.5, 0.1]))
    m2 = SimpleNamespace(predict=lambda X: np.array([0.1, 0.8, 0.9]))
    df_f = df.assign(**{f: 0.0 for f in FEATURES})
    assert predict_score([m1, m2], df_f).tolist() == [-2.0, -2.0, -2.0]
    # ties in score inside one model -> candidate order: (0.5, 0.5, 0.9) -> ranks (2, 3, 1)
    assert within_user_rank(df, np.array([0.5, 0.5, 0.9])).tolist() == [2, 3, 1]
