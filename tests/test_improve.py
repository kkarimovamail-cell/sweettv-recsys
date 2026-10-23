import math

import numpy as np
import pandas as pd

from src.improve import airing_weight, neighbours
from src.ppr import interaction_matrices


def train_rows():
    # u watched "1" in week 20 and week 18, "2" in week 16; end of train window = week 20
    return pd.DataFrame({"user_id": ["u", "u", "u", "u"], "tv_show_id": ["1", "1", "2", "3"],
                         "overlap_s": [1, 1, 1, 1], "watched": [True, True, True, False], "week": [20, 18, 16, 20]})


def test_recency_infinite_half_life_is_binary_and_counts():
    X, H, _, items = interaction_matrices(train_rows())
    assert items == ["1", "2"]
    assert X.toarray().tolist() == [[1.0, 1.0]] and H.toarray().tolist() == [[2.0, 1.0]]


def test_recency_half_life_two_weeks_by_hand():
    # H = 2: week 20 -> 0.5^0 = 1, week 18 -> 0.5^1 = 0.5, week 16 -> 0.5^2 = 0.25
    # X = most recent weight: "1" -> 1, "2" -> 0.25;  H = sum: "1" -> 1.5, "2" -> 0.25
    X, H, _, _ = interaction_matrices(train_rows(), half_life=2.0, end_week=20)
    np.testing.assert_allclose(X.toarray(), [[1.0, 0.25]])
    np.testing.assert_allclose(H.toarray(), [[1.5, 0.25]])


def test_airing_weight_beta():
    # n_airings in weeks 21-30: "1" -> 4, "2" -> 9, "7" not in items -> 0
    items = pd.DataFrame({"tv_show_id": ["1", "2"], "n_airings_w21_30": [4, 9], "n_airings_w31_42": [0, 0]})
    graph_items = np.array(["1", "2", "7"])
    np.testing.assert_allclose(airing_weight(items, graph_items, "val", 0.5), [2.0, 3.0, 0.0])
    np.testing.assert_allclose(airing_weight(items, graph_items, "val", 0.0), [1.0, 1.0, 1.0])


def test_neighbours_stay_in_grid():
    assert neighbours([0.0, 0.25, 0.5], 0.0) == [0.0, 0.25]
    assert neighbours([math.inf, 8.0, 4.0, 2.0], 4.0) == [8.0, 4.0, 2.0]
    assert neighbours([math.inf, 8.0, 4.0, 2.0], math.inf) == [math.inf, 8.0]


def test_infinite_half_life_through_formula_equals_no_decay():
    a = interaction_matrices(train_rows())
    b = interaction_matrices(train_rows(), half_life=math.inf, end_week=20)
    assert (a[0] != b[0]).nnz == 0 and (a[1] != b[1]).nnz == 0


def test_soft_signal_weights_by_hand():
    # signal 0.5 on the week-20 airing of "1", 1.0 elsewhere; H = 2, end week 20
    # "1": weights 1*0.5 = 0.5 (week 20) and 0.5*1.0 = 0.5 (week 18) -> X = max 0.5, H = 1.0
    # "2": 0.25 * 1.0 -> X = H = 0.25
    rows = train_rows().assign(signal=[0.5, 1.0, 1.0, 0.0])
    X, H, _, _ = interaction_matrices(rows, half_life=2.0, end_week=20)
    np.testing.assert_allclose(X.toarray(), [[0.5, 0.25]])
    np.testing.assert_allclose(H.toarray(), [[1.0, 0.25]])
