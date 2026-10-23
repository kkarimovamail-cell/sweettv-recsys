import numpy as np
import pandas as pd
import pytest

from src import audit, config


def test_near_boundary_tolerance():
    bounds = {7: np.array([1_000, 5_000])}
    t = np.array([1_000, 1_120, 1_121, 4_880, 3_000])
    hit = audit.near_boundary(np.full(len(t), 7), t, bounds)
    assert hit.tolist() == [True, True, False, True, False]


def test_date_without_format_is_order_dependent():
    values = pd.Series(["09.03.2020 06:00:00", "13.03.2020 00:50:00"])
    assert audit.try_vectorised(values)["error"] is not None
    correct = pd.to_datetime(values, format=audit.SCHEDULE_DT_FORMAT)
    assert correct.dt.month.tolist() == [3, 3] and correct.dt.day.tolist() == [9, 13]


@pytest.mark.skipif(not (config.INTERIM_DIR / "log.parquet").exists(), reason="run python -m src.audit first")
def test_cache_dtypes():
    log = pd.read_parquet(config.INTERIM_DIR / "log.parquet")
    assert log["channel_id"].dtype == "int32" and log["user_id"].dtype == "str"
    for period in config.SCHEDULE_FILES:
        sched = pd.read_parquet(config.INTERIM_DIR / f"schedule_{period}.parquet", columns=["channel_id", "tv_show_id"])
        assert sched["channel_id"].dtype == "int32" and sched["tv_show_id"].dtype == "str"
