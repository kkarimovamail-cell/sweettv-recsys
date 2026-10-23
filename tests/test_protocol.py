import pandas as pd

from src import protocol as P


def views(rows):  # rows: (user, tv_show_id, overlap_s, watched, week)
    return pd.DataFrame(rows, columns=["user_id", "tv_show_id", "overlap_s", "watched", "week"])


def test_rank_tie_rule_count_then_overlap_then_numeric_id():
    # 7 (2 airings) > 10 (1 airing, overlap 900) > 3 and 9 (1 airing, 600 each) -> 3 before 9;
    # ids compare as numbers, so "10" is not placed before "3" by string order
    av = views([("u", "7", 100, True, 21), ("u", "7", 100, True, 22),
                ("u", "10", 900, True, 21), ("u", "9", 600, True, 21), ("u", "3", 600, True, 21),
                ("u", "5", 3000, False, 21)])  # not watched -> ignored
    ranked = P.to_lists(P.rank_per_user(P.watched_counts(av)))
    assert ranked == {"u": ["7", "10", "3", "9"]}


def test_truth_needs_five_programmes_and_uses_target_weeks():
    # u: 5 programmes in weeks 21-30 -> truth; v: 4 in target + 1 in train week 20 -> no truth
    rows = [("u", str(i), 100, True, 25) for i in range(1, 6)]
    rows += [("v", str(i), 100, True, 25) for i in range(1, 5)] + [("v", "9", 100, True, 20)]
    truth = P.build_truth(views(rows), "val")
    assert truth == {"u": ["1", "2", "3", "4", "5"]}
    # without weeks 25-27 u has nothing left
    assert P.build_truth(views(rows), "val", exclude_weeks=(25, 27)) == {}


def test_popularity_counts_distinct_users_within_pool():
    # "1": users u, v (u twice) -> 2; "2": user w -> 1; "3": 3 users but outside the pool
    av = views([("u", "1", 1, True, 11), ("u", "1", 1, True, 12), ("v", "1", 1, True, 11),
                ("w", "2", 1, True, 11), ("u", "3", 1, True, 11), ("v", "3", 1, True, 11),
                ("w", "3", 1, True, 11)])
    assert P.popularity_ranking(av, pool={"1", "2"}) == ["1", "2"]


def test_fill_to_k_dedups_filters_pool_and_tops_up():
    pool = {"1", "2", "3", "4", "5", "6", "7"}
    popular = ["1", "2", "3", "4", "5", "6"]
    # u: own ["6", "6", "99"] -> "99" not in pool, repeat dropped -> ["6"] + 1,2,3,4
    # c: cold start -> 1,2,3,4,5
    preds, stats = P.fill_to_k({"u": ["6", "6", "99"]}, ["u", "c"], popular, pool)
    assert preds == {"u": ["6", "1", "2", "3", "4"], "c": ["1", "2", "3", "4", "5"]}
    assert stats == {"users": 2, "users_without_own_ids": 1, "users_topped_up": 2, "slots_from_popularity": 9}


def test_evaluate_oracle_is_one():
    truth = {"u": ["1", "2", "3", "4", "5"], "v": ["6", "7", "8", "9", "10"]}
    assert P.evaluate(truth, truth, {"A": ["u", "v"]}) == {"map@5_A": 1.0}
