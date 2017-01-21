import numpy as np


def to_numpy(arr):
    if type(arr) is not np.ndarray:
        return np.asarray(arr)
    return arr

def l2_distance(x, y):
    x = to_numpy(x)
    y = to_numpy(y)
    return np.linalg.norm(x-y)
