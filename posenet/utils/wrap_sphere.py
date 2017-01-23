import random

import numpy as np

from posenet.utils.coordinates import to_cartesian
from posenet.utils.poisson_sampler import PoissonSampler


def wrap_on_sphere(u, v, phi1, phi2, theta1, theta2):
    phi = (phi2 - phi1) * u + phi1
    theta = np.arccos(np.cos(theta1) + (np.cos(theta2) -np.cos(theta1))*v)
    return phi, theta

def sample_square(n):
    '''Take n samples from poisson distribution between 0 and 1'''

    # Sample
    r = 0.86 / np.sqrt(n) # Assume density
    sampler = PoissonSampler([0,1], [0,1], r)
    samples = sampler.sample()

    x = np.array([s[0] for s in samples])
    y = np.array([s[1] for s in samples])
    return x, y

def sample_spherical(n, phi1=0, phi2=2*np.pi, theta1=0, theta2=np.pi):
    u, v = sample_square(n)
    phi, theta = wrap_on_sphere(u, v, phi1, phi2, theta1, theta2)
    x, y, z = to_cartesian(phi, theta)
    return x, y, z

def sample_cap(n, cap_phi, cap_theta, cap_alpha):
    u, v = sample_square(n)
    phi, theta = wrap_on_sphere(u, v, phi1=0, phi2=2*np.pi, theta1=0, theta2=cap_alpha)
    x, y, z = to_cartesian(phi, theta)
    cx, cy, cz = to_cartesian(cap_phi, cap_theta)

    # Find rotation from [x,y,z] to [cx,cy,cz]
    angle = np.arccos(np.array([0, 0, 1]).dot([cx, cy, cz]))
    if abs(angle - np.pi) < 0.001:
        k = np.array([0,1,0]) # Choose y axis to rotate
    else:
        k = np.cross(np.array([0, 0, 1]), np.array([cx, cy, cz]))
        k = k / np.linalg.norm(k)
    
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    R = np.eye(3) + np.sin(angle)*K + (1 - np.cos(angle)) * K.dot(K)

    # Rotate everything
    x, y, z = np.dot(R, np.stack([x, y, z], axis=0))

    return x, y, z