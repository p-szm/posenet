import numpy as np

def to_cartesian(phi, theta):
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    return x, y, z

def to_spherical(x, y, z):
	r = np.sqrt(x**2 + y**2 + z**2)
	phi = np.arctan2(y, x)
	theta = np.arccos(float(z)/r)
	return r, phi, theta