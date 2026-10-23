import pytest

from src.metrics import apk, mapk


@pytest.mark.parametrize("actual, predicted, expected", [
    # all five hits: (1/1 + 2/2 + 3/3 + 4/4 + 5/5) / 5 = 1.0
    (["1", "2", "3", "4", "5"], ["1", "2", "3", "4", "5"], 1.0),
    # hits at positions 2 and 4: (1/2 + 2/4) / 5 = 0.2
    (["1", "2", "3", "4", "5"], ["6", "1", "7", "2", "8"], 0.2),
    # only two relevant -> divide by min(2, 5) = 2: hits at 1 and 5: (1/1 + 2/5) / 2 = 0.7
    (["1", "2"], ["2", "6", "7", "8", "1"], 0.7),
    # repeated prediction counts once: hit at 1, repeat at 2 skipped, hit at 3: (1/1 + 2/3) / 2 = 0.8333
    (["1", "2"], ["1", "1", "2", "8", "9"], (1 + 2 / 3) / 2),
    # hit only beyond k is cut off: 0.0
    (["6"], ["1", "2", "3", "4", "5", "6"], 0.0),
    # empty ground truth -> 0.0
    ([], ["1", "2", "3", "4", "5"], 0.0),
])
def test_apk(actual, predicted, expected):
    assert apk(actual, predicted, k=5) == pytest.approx(expected)


def test_mapk_is_mean_of_apk():
    # (1.0 + 0.2) / 2 = 0.6
    actual = [["1", "2", "3", "4", "5"], ["1", "2", "3", "4", "5"]]
    predicted = [["1", "2", "3", "4", "5"], ["6", "1", "7", "2", "8"]]
    assert mapk(actual, predicted, k=5) == pytest.approx(0.6)
