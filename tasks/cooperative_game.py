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
    Returns the Shapley value based on the simplified formula.

    Parameters:
        P (list): All players.
        j (int): Selected player. j is in the list of P.
        v (func): Function to compute the value given a set/list.
            Elements from set/list must be in P.

    Returns:
        value (float): Rounded Shapley value to two decimal places.
        S_all (list of sets): One set for each permutation order. Set should only contain the players
            before j is added.
    """

    S_all = [set(s) for s in itertools.chain.from_iterable(itertools.combinations(P, r) for r in range(len(P) + 1))]
    value = 0.0

    for S in S_all:
        if j in S:
            continue
        cardinality = len(S)
        # Using the simplified formula for Shapley value
        value += (factorial(cardinality) * factorial(len(P) - cardinality - 1) / factorial(len(P)) *
                  (v(S.union({j})) - v(S)))

    # Convert the result to match the format of get_shapley_by_order
    S_all = [S for S in S_all if j not in S]
    value = round(value, 2)

    return value, S_all


def get_shapley_by_order(P, j, v, M=None):
    """
    Returns the Shapley value based on order permutations.

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

    # Initialize a list to store sets for each permutation order
    S_all = [set() for _ in range(math.factorial(len(P)))]

    # Initialize the Shapley value
    value = 0.0

    # Generate all possible permutations of players
    permutations = list(itertools.permutations(P))

    # If M is specified, truncate the list of permutations
    if M is not None:
        permutations = random.sample(permutations, min(M, len(permutations)))

    # Iterate over all permutations
    for i, perm in enumerate(permutations):
        # Find the index of the selected player in the permutation
        index_j = perm.index(j)

        # Create a set containing players before j is added
        S_all[i] = set(perm[:index_j])

        # Update the Shapley value based on the difference in values
        value += (v(S_all[i].union({j})) - v(S_all[i]))

    # Normalize the Shapley value by dividing by the factorial of the number of players
    value /= math.factorial(len(P))
    value = round(value, 2)

    # Round the result to two decimal places
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
