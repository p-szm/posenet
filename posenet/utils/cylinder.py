import random
import numpy as np


def cylinder_points(r, n_theta, z_min, z_max, d=None, 
        theta_min=0, theta_max=2*np.pi, add_last=False):
    points = []
    z = z_min
    k = 1 if add_last else 0
    while z <= z_max:
        thetas = theta_min + np.arange(n_theta + k) * (theta_max - theta_min) / n_theta
        for theta in thetas:
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            points.append([x, y, z])
        if d is None:
            z += (theta_max - theta_min) * r / n_theta
        else:
            z += d
    return np.array(points)

def random_cylinder_points(n, r_min, r_max, z_min, z_max, 
        theta_min=0, theta_max=2*np.pi):
    points = []
    for i in range(n):
        r = random.uniform(r_min, r_max)
        theta = random.uniform(theta_min, theta_max)
        z = random.uniform(z_min, z_max)
        points.append([r*np.cos(theta), r*np.sin(theta), z])
    return np.array(points)