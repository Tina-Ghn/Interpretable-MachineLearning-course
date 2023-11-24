def get_value(S, comb):
    """
    In our case a simple look-up.

    Parameters:
        S (set or list): Set on which the cost should be evaluated. If list is given, list is
        converted to set.
    """

    if type(S) is list:
        S = set(S)

    for key, value in comb.items():

        if type(key) == int:
            key = [key]

        if set(key) == S:
            return value
