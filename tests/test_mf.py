import numpy as np
import pandas as pd
import scipy.sparse as sp

from src.mf import fit_als, recommend, user_item_matrix


def test_user_item_matrix_counts_watched_airings():
    # u: "10" watched twice, "9" once, "7" not watched -> row [9: 1, 10: 2]; items sorted numerically
    train = pd.DataFrame({"user_id": ["u", "u", "u", "u"], "tv_show_id": ["10", "10", "9", "7"],
                          "overlap_s": [1, 1, 1, 1], "watched": [True, True, True, False], "week": [11] * 4})
    M, users, items = user_item_matrix(train)
    assert items.tolist() == ["9", "10"] and users.tolist() == ["u"]
    assert M.toarray().tolist() == [[1.0, 2.0]]


def test_recommend_keeps_watched_masks_pool_and_breaks_ties_by_id():
    # one user, factor 1.0; item scores = item factors: 1 -> 5, 2 -> 4, 3 -> 9 (not in pool), 4 -> 4, 5 -> 1, 6 -> 0
    # pool order by score: 1 (5), 2 and 4 tie at 4 -> 2 first, 5 (1), 6 (0)
    items = np.array(["1", "2", "3", "4", "5", "6"])
    item_f = np.array([[5.0], [4.0], [9.0], [4.0], [1.0], [0.0]])
    preds, fill = recommend(np.array([[1.0]]), item_f, pd.Index(["u"]), items,
                            pool={"1", "2", "4", "5", "6", "8"}, users=["u", "cold"], popular=["8", "1", "2", "4", "5"])
    assert preds["u"] == ["1", "2", "4", "5", "6"]
    assert preds["cold"] == ["8", "1", "2", "4", "5"]  # cold start -> popularity
    assert fill["users_without_own_ids"] == 1


def test_als_is_deterministic_with_fixed_seed():
    M = sp.random(60, 40, density=0.1, format="csr", random_state=42, dtype=np.float32)
    a, b = fit_als(M, 8, 0.1, 10.0), fit_als(M, 8, 0.1, 10.0)
    assert np.array_equal(a.user_factors, b.user_factors) and np.array_equal(a.item_factors, b.item_factors)
