import math

import numpy as np
import pandas as pd

from src.cbf import fuse, item_matrix, split_people, token_block


def test_idf_by_hand():
    # 3 programmes; person "a" in 2 of them, "b" in 1:
    # idf(a) = ln(4/3) + 1 = 1.2877, idf(b) = ln(4/2) + 1 = 1.6931; row 0 = [a, b] -> L2-normalised
    X = token_block([["a", "b"], ["a"], []], idf=True).toarray()
    ia, ib = math.log(4 / 3) + 1, math.log(2) + 1
    np.testing.assert_allclose(X[0], np.array([ia, ib]) / math.hypot(ia, ib))
    np.testing.assert_allclose(X[1], [1.0, 0.0])
    np.testing.assert_allclose(X[2], [0.0, 0.0])  # empty block stays zero


def test_split_people():
    assert split_people("Том Круз, Сара Райт,  Том Круз") == ["Сара Райт", "Том Круз"]
    assert split_people(None) == [] and split_people(float("nan")) == []


def test_item_vector_blocks_and_cosine():
    # "1" and "2" share category and genre, differ in actors; "3" shares nothing with "1"
    items = pd.DataFrame({
        "tv_show_id": ["2", "10", "1"], "tv_show_category": ["Фильмы", "Инфо", "Фильмы"],
        "tv_show_genre_1": ["драма", None, "драма"], "tv_show_genre_2": [None, None, None], "tv_show_genre_3": [None, None, None],
        "director": [None, None, None], "actors": ["X", None, "Y"]})
    V, ids, stats = item_matrix(items)
    assert ids.tolist() == ["1", "2", "10"]  # numeric order
    # each of "1", "2": blocks category, genre, actors each unit length -> vector norm sqrt(3) before final L2
    # cos("1", "2") = (1 + 1 + 0) / 3 = 0.6667; cos("1", "10") = 0
    sim = (V @ V.T).toarray()
    assert abs(sim[0, 1] - 2 / 3) < 1e-12 and sim[0, 2] == 0
    assert stats["actors"]["programmes_with_any"] == 2


def test_fuse_mu_zero_is_v2_and_mu_adds_cbf():
    v2r, v2s = {"u": ["a", "b"]}, {"u": np.array([2.0, 1.0])}   # normalised 1.0, 0.5
    cr, cs = {"u": ["c", "b"]}, {"u": np.array([0.8, 0.4])}     # normalised 1.0, 0.5
    assert fuse(v2r, v2s, cr, cs, 0.0) == {"u": ["a", "b"]}   # c scores 0 -> dropped
    # mu = 1: a 1.0, b 0.5 + 0.5 = 1.0, c 1.0 -> three-way tie keeps candidate order a, b, c
    assert fuse(v2r, v2s, cr, cs, 1.0) == {"u": ["a", "b", "c"]}
    # mu = 0.25: a 1.0, b 0.625, c 0.25
    assert fuse(v2r, v2s, cr, cs, 0.25) == {"u": ["a", "b", "c"]}
