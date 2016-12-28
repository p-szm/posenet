import math

import numpy as np


def l2_distance(x, y):
	if type(x) is not np.ndarray:
		x = np.asarray(x)
	if type(y) is not np.ndarray:
		y = np.asarray(y)
	return np.linalg.norm(x-y)

def quaternion_distance(q1, q2):
	"""Returns an angle that rotates q1 into q2"""
	if type(q1) is not np.ndarray:
		q1 = np.asarray(q1)
	if type(q2) is not np.ndarray:
		q2 = np.asarray(q2)
	cos = 2*(q1.dot(q2))**2 - 1
	cos = max(min(cos, 1), -1) # To combat numerical imprecisions
	return math.acos(cos)