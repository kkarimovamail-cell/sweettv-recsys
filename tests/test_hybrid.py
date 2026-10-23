import pytest

from src.hybrid import rrf

PPR = {"u": ["1", "2", "3"]}
MF = {"u": ["3", "4"]}


def test_w1_is_ppr_and_w0_is_mf():
    # zero-score items are dropped, so the other method contributes nothing
    assert rrf(PPR, MF, ["u"], 1.0) == {"u": ["1", "2", "3"]}
    assert rrf(PPR, MF, ["u"], 0.0) == {"u": ["3", "4"]}


def test_half_weight_scores_by_hand():
    # w = 0.5, k = 60:
    # "3": 0.5/63 + 0.5/61 = 0.016133   "1": 0.5/61 = 0.008197   "4": 0.5/62 = 0.008065   "2": 0.5/62 = 0.008065
    # "2" and "4" tie -> smaller id first
    assert rrf(PPR, MF, ["u"], 0.5) == {"u": ["3", "1", "2", "4"]}
    assert 0.5 / 63 + 0.5 / 61 == pytest.approx(0.016133, abs=1e-6)


def test_user_missing_in_one_method():
    # v is unknown to MF: at w = 0.5 only PPR's item remains;
    # at w = 0 that item scores 0 and is dropped (popularity fills later)
    assert rrf({"v": ["9"]}, {}, ["v"], 0.5) == {"v": ["9"]}
    assert rrf({"v": ["9"]}, {}, ["v"], 0.0) == {"v": []}
