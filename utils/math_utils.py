import numpy as np


def mean(list_val, fallback_val=None):
    if len(list_val) == 0:
        if fallback_val is not None:
            return fallback_val
        else:
            raise ZeroDivisionError()
    return sum(list_val) / len(list_val)


def variance(list_val, fallback_val=0):
    if len(list_val) == 0:
        if fallback_val is not None:
            return fallback_val
        else:
            raise ValueError()
    v = np.var(np.asarray(list_val))
    return v


def median(list_val, fallback_val=0):
    if len(list_val) == 0:
        if fallback_val is not None:
            return fallback_val
        else:
            raise ValueError()
    v = np.median(np.asarray(list_val))
    return v


def make_divisible(v, divisor, min_val=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_val:
    :return:
    """
    if min_val is None:
        min_val = divisor
    new_v = max(min_val, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

