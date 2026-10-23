import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from src.ppr import coview_transition, interaction_matrices, ppr


def X_from(rows, n_items):
    r, c = zip(*rows)
    return sp.csr_matrix((np.ones(len(rows)), (r, c)), shape=(max(r) + 1, n_items))


def test_coview_counts_rows_and_dangling():
    # users: u0{a,b}, u1{a,b}, u2{b,c}, u3{d} -> W: a-b 2, b-c 1, d has no neighbour
    # P rows: a -> b 1.0; b -> a 2/3, c 1/3; c -> b 1.0; d dangling
    X = X_from([(0, 0), (0, 1), (1, 0), (1, 1), (2, 1), (2, 2), (3, 3)], 4)
    Pm, dangling, stats = coview_transition(X, top_k=50)
    expected = np.array([[0, 1, 0, 0], [2 / 3, 0, 1 / 3, 0], [0, 1, 0, 0], [0, 0, 0, 0]])
    np.testing.assert_allclose(Pm.toarray(), expected)
    assert dangling.tolist() == [False, False, False, True]
    assert stats["max_abs_row_sum_minus_1"] < 1e-12


def test_top_k_keeps_heaviest_and_breaks_ties_by_smaller_id():
    # b's neighbours: a (2), c (1), d (1); top_k=2 keeps a and c (c < d) -> b: a 2/3, c 1/3
    X = X_from([(0, 0), (0, 1), (1, 0), (1, 1), (2, 1), (2, 2), (3, 1), (3, 3)], 4)
    Pm, _, _ = coview_transition(X, top_k=2)
    np.testing.assert_allclose(Pm.toarray()[1], [2 / 3, 0, 1 / 3, 0])


def test_ppr_two_node_cycle_closed_form():
    # a <-> b, v = a, alpha 0.5: p_a = 0.5 p_b + 0.5, p_b = 0.5 p_a -> p = (2/3, 1/3)
    PT = sp.csr_matrix(np.array([[0, 1], [1, 0]], dtype=float)).T.tocsr()
    p, it, conv, mass = ppr(PT, np.array([False, False]), np.array([[1.0], [0.0]]), 0.5)
    np.testing.assert_allclose(p.ravel(), [2 / 3, 1 / 3], atol=1e-6)
    assert conv and it < 200 and mass < 1e-12


def test_ppr_dangling_mass_returns_to_restart():
    # a -> b, b dangling, v = a, alpha 0.5: p_b = 0.5 p_a; p_a = 0.5 * d + 0.5 with d = p_b -> (2/3, 1/3)
    PT = sp.csr_matrix(np.array([[0, 1], [0, 0]], dtype=float)).T.tocsr()
    p, _, conv, mass = ppr(PT, np.array([False, True]), np.array([[1.0], [0.0]]), 0.5)
    np.testing.assert_allclose(p.ravel(), [2 / 3, 1 / 3], atol=1e-6)
    assert conv and abs(p.sum() - 1) < 1e-12 and mass < 1e-12


def test_alpha_zero_returns_restart():
    PT = sp.csr_matrix(np.array([[0, 1], [1, 0]], dtype=float)).T.tocsr()
    V = np.array([[0.25], [0.75]])
    p, it, conv, _ = ppr(PT, np.array([False, False]), V, 0.0)
    # alpha = 0: p = v after one step, delta = 0 -> converged at iteration 1
    np.testing.assert_allclose(p, V)
    assert conv and it == 1


def test_interaction_matrices_sort_items_numerically():
    train = pd.DataFrame({"user_id": ["u", "u", "u"], "tv_show_id": ["10", "9", "10"],
                          "overlap_s": [1, 1, 1], "watched": [True, True, True], "week": [11, 11, 12]})
    X, H, users, items = interaction_matrices(train)
    assert items == ["9", "10"] and users == ["u"]
    assert X.toarray().tolist() == [[1.0, 1.0]] and H.toarray().tolist() == [[1.0, 2.0]]
