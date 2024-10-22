def update_mean(m: float, x: float, k: int) -> float:
    """
    Update the mean using the recursive algorithm.

    Args:
        m (float): Current mean.
        x (float): New value.
        k (int): Number of values seen so far.

    Returns:
        float: Updated mean.

    Reference:
        https://math.stackexchange.com/a/1836447
    """
    return m + (x - m) / k


def update_variance(x: float, m: float, var: float, k: int) -> float:
    """
    Update the variance using Welford's online algorithm.

    Args:
        x (float): New value.
        m (float): Current mean.
        var (float): Current variance.
        k (int): Number of values seen so far.

    Returns:
        float: Updated variance.

    Reference:
        https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm
    """
    delta = x - m
    return var + delta * (x - (m + delta / k))
