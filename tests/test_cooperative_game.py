import sys
import os  # noqa
sys.path.insert(0, "")  # noqa

from utils.shapley import get_value
from datasets.shapley import COMB1, COMB2, COMB3, COMB4, COMB5, COMB6, COMB7, COMB8, COMB9, COMB10, COMB11, COMB12, COMB13, COMB14, COMB15, COMB16
from tests.config import WORKING_DIR

module = __import__(f"{WORKING_DIR}.cooperative_game", fromlist=[
    'get_shapley', 'get_shapley_by_order', 'check_symmetry', 'check_dummy', 'check_additivity', 'check_efficiency'])


def test_get_shapley():
    P = [1, 2, 3, 4]
    def v(S): return get_value(S, COMB10)
    value, S_all = module.get_shapley(P, P[1], v)

    assert value == 20.75
    assert set(()) in S_all
    assert set((1, 4)) in S_all


def test_get_shapley_by_order():
    P = [1, 2, 3, 4]
    def v(S): return get_value(S, COMB10)

    value, S_all = module.get_shapley_by_order(P, P[3], v)

    assert value == 31.25

    count_2 = 0
    count_3 = 0
    for S in S_all:
        if S == set((2, 3)):
            count_2 += 1

        if S == set((1, 2, 3)):
            count_3 += 1

    assert count_2 == 2
    assert count_3 == 6


def test_get_shapley_by_order_approx():
    P = [1, 2, 3, 4]
    def v(S): return get_value(S, COMB10)

    value, S_all = module.get_shapley_by_order(P, P[3], v, M=24)

    assert value == 31.25

    count_2 = 0
    count_3 = 0
    for S in S_all:
        if S == set((2, 3)):
            count_2 += 1

        if S == set((1, 2, 3)):
            count_3 += 1

    assert count_2 == 2
    assert count_3 == 6

    value, S_all = module.get_shapley_by_order(P, P[3], v, M=10)

    assert value == 44.7


if __name__ == "__main__":
    test_get_shapley()
    test_get_shapley_by_order()
    test_get_shapley_by_order_approx()
