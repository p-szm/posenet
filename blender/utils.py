import math
import random

import bpy
from mathutils import *
import numpy as np

from poisson_sampler import PoissonSampler


def preprocess_args(argv):
    return argv[argv.index('--') + 1:] if '--' in argv else []

def renderToFile(filename, width, height, mode='RGB'):
    print(width, height)
    """Save the scene as a png file"""
    bpy.data.scenes['Scene'].render.filepath = filename
    bpy.context.scene.render.resolution_percentage = 100
    bpy.context.scene.render.resolution_x = width 
    bpy.context.scene.render.resolution_y = height
    bpy.context.scene.render.image_settings.color_mode = mode
    bpy.ops.render.render(write_still=True)

class Camera:
    def __init__(self, camera):
        self.camera = camera

    def setLocation(self, r):
        if type(r) is not Vector:
            r = Vector(r)
        self.camera.location = r

    def setRotation(self, q):
        if type(q) is not Quaternion:
            q = Quaternion(q)
        self.camera.rotation_euler = q.to_euler()

    def look_at(self, point):
        # Point the camera's '-Z' and use its 'Y' as up
        q = (point - self.camera.location).to_track_quat('-Z', 'Y')
        self.camera.rotation_euler = q.to_euler()

    def getLocation(self):
        return self.camera.location

    def getRotation(self):
        return self.camera.rotation_euler.to_quaternion()

    def getPoseString(self, decimals=6):
        r = [round(v, decimals) for v in self.getLocation()]
        q = [round(v, decimals) for v in self.getRotation()]
        return '{} {} {} {} {} {} {}'.format(r[0], r[1], r[2], q[0], q[1], q[2], q[3])

def wrap_on_sphere(u, v, phi1, phi2, theta1, theta2):
    phi = (phi2 - phi1) * u + phi1
    theta = np.arccos(np.cos(theta1) + (np.cos(theta2) -np.cos(theta1))*v)
    return phi, theta

def to_cartesian(phi, theta):
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    return x, y, z

def sample_square(n):
    '''Take n samples from poisson distribution between 0 and 1'''

    # Sample
    r = 0.86 / math.sqrt(n) # Assume density
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
    # Doesn't work for cap_theta=np.pi

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
