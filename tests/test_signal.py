import pandas as pd

from src.signal_exp import signal_train


def views():
    return pd.DataFrame({"user_id": ["u"] * 4, "tv_show_id": ["1", "2", "3", "4"], "overlap_s": [1] * 4,
                         "watched": [True, False, False, False], "week": [11, 11, 11, 25],
                         "screen_time": [0.9, 0.6, 0.4, 0.95]})


def test_threshold_and_soft_signal():
    # train weeks 11-20 only (week-25 row dropped)
    t = signal_train(views(), "val", "threshold", 0.5)
    assert t["watched"].tolist() == [True, True, False]
    s = signal_train(views(), "val", "soft", 0.3)
    assert s["watched"].tolist() == [True, True, True] and s["signal"].tolist() == [0.9, 0.6, 0.4]
    # T = 0.8 reproduces the stored watched flag
    assert signal_train(views(), "val", "threshold", 0.8)["watched"].tolist() == [True, False, False]
