import sys
from time import sleep

import numpy as np


def to_numpy(arr):
    if type(arr) is not np.ndarray:
        return np.asarray(arr)
    return arr

def l2_distance(x, y):
    x = to_numpy(x)
    y = to_numpy(y)
    return np.linalg.norm(x-y)

def progress_bar(r, width):
	k = int(round(r * width))
	bar = '[' + '-'*k + ' '*(width-k) + '] ' + str(round(100*r,1)) + '%'
	sys.stdout.write('\r')
	sys.stdout.write(bar)
	sys.stdout.flush()