import numpy as np
import pandas as pd
import scipy.sparse as sp

from src.expected import expected_table, rerank


def test_expected_by_hand():
    # user u, graph items ["1", "2"], recency-weighted watched: "1" -> 3.0, "2" -> 0
    # train airings: "1" -> 6, "2" -> 4, "9" -> 0;  target airings: "1" -> 10, "2" -> 5, "9" -> 2
    # expected: "1" = 3/6 * 10 = 5.0; "2" = 0; "9" (never watched, not in graph) = 0
    graph = {"items": np.array(["1", "2"]), "users": pd.Index(["u"]), "H": sp.csr_matrix([[3.0, 0.0]])}
    items = pd.DataFrame({"tv_show_id": ["1", "2", "9"], "n_airings_w11_20": [6, 4, 0],
                          "n_airings_w21_30": [0, 0, 0], "n_airings_w31_42": [0, 0, 0]})
    items["n_airings_w21_30"] = [10, 5, 2]
    out = expected_table(graph, items, "val", {"u": ["2", "9", "1"]})
    np.testing.assert_allclose(out["u"], [0.0, 0.0, 5.0])


def test_rerank_lambda_and_ppr_only():
    ranked = {"u": ["a", "b", "c", "d", "e", "f"]}
    ppr = {"u": np.array([6.0, 5.0, 4.0, 3.0, 2.0, 1.0])}  # normalised: 1, 5/6, 4/6, 3/6, 2/6, 1/6
    exp = {"u": np.array([0.0, 0.0, 2.0, 0.0, 2.0, 1.0])}
    # ppr only -> PPR order
    assert rerank(ranked, ppr, None, 0.0) == {"u": ["a", "b", "c", "d", "e"]}
    # lambda 0: c and e tie at 2 -> PPR order (c first); f 1; then a, b, d tie at 0 -> PPR order
    assert rerank(ranked, ppr, exp, 0.0) == {"u": ["c", "e", "f", "a", "b"]}
    # lambda 1: a 1.0, b 0.833, c 2.667, d 0.5, e 2.333, f 1.167 -> c, e, f, a, b
    assert rerank(ranked, ppr, exp, 1.0) == {"u": ["c", "e", "f", "a", "b"]}
