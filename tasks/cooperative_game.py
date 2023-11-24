import math
import random
import sys
import os  # noqa

sys.path.insert(0, "")  # noqa

from utils.shapley import get_value
from datasets.shapley import COMB1, COMB3, COMB13, COMB14, COMB15, COMB16
import itertools
import numpy as np
from math import factorial


def get_shapley(P, j, v):
    """
    Returns the shapley value based on the original implementation.

    Parameters:
        P (list): All players.
        j (int): Selected player. j is in the list of P.
        v (func): Function to compute the value given a set/list.
            Elements from set/list must be in P.

    Returns:
        value (float): Rounded Shapley value to two decimal places.
        S_all (list of sets): All possible combinations.
    """

    S_all = #TODO
    value = #TODO
    return value, S_all


def get_shapley_by_order(P, j, v, M=None):
    """
    Returns the shapley value based on order permutations.

    Parameters:
        P (list): All players.
        j (int): Selected player. j is in the list of P.
        v (func): Function to compute the value given a set/list.
            Elements from set/list must be in P.
        M (int): Number of used permutations. Optional.

    Returns:
        value (float): Rounded Shapley value to two decimal places.
        S_all (list of sets): One set for each permutation order. Set should only contain the players
            before j is added.
    """

    S_all = #TODO
    value = #TODO

    return value, S_all


if __name__ == "__main__":

    S = [1, 3]
    P = [1, 2, 3]

    def v(S):
        return get_value(S, COMB1)

    with open("coop_game.txt", "w") as f:
        f.write("Given are the following sets:\n")
        f.write(f"{str(COMB1)}\n")

        f.write("\nShapley by original implementation:\n")
        for j in P:
            shapley = get_shapley(P, j, v)
            f.write(f"--- P{j}: {shapley[0]}\n")

        f.write("\nShapley by orders:\n")
        for j in P:
            shapley = get_shapley_by_order(P, j, v)
            f.write(f"--- P{j}: {shapley[0]}\n")