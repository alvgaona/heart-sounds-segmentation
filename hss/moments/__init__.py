def update_mean(m: float, x: float, k: int):
    """
    Step to update the mean using the recursive algorithm in:
    https://math.stackexchange.com/a/1836447
    """
    m += (x - m) / k
    return m


def update_variance(x: float, m: float, var: float):
    """
    Step to update the variance using the Welford's method.
    See: https://math.stackexchange.com/a/775678
    """
    var += (x - m) * (x - m)

    return var
