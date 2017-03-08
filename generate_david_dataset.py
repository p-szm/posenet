import math
import os
import random
import sys
import time
import numpy as np

sys.path.append(os.path.dirname(__file__)) # So that next imports work
from posenet.blender import *
from posenet.utils import *


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


dataset_type = 'extrapolation'
mode = 'training'
output_dir = 'datasets/david'
size = [512, 512]
dv = 0
r = 3.5

if dataset_type == 'interpolation':
    if mode == 'training':
        dataset_name = 'training'
        n_theta = 20
        d = 2 * np.pi * r / n_theta
        points = cylinder_points(r, n_theta, 0.5, 6.5)
        points_in = cylinder_points(r-d, n_theta, 0.5, 6.5, d=d)
        points_out = cylinder_points(r+d, n_theta, 0.5, 6.5, d=d)
        points = np.concatenate([points_in, points, points_out], axis=0)
    elif mode == 'validation':
        dataset_name = 'validation'
        points = random_cylinder_points(90, 2.2, 4.8, 0.5, 6.5)
    elif mode == 'test1':
        dataset_name = 'test1'
        points = cylinder_points(r, 200, 1.599557, 1.599557)
    elif mode == 'test2':
        dataset_name = 'test2'
        n = 5*20 + 1
        x = np.linspace(1.08156, 1.08156, n)
        y = np.linspace(-3.328698, -3.328698, n)
        z = np.linspace(0.5, 5.997787, n)
        points = np.column_stack((x, y, z))
    elif mode == 'dev':
        dataset_name = 'dev'
        points = cylinder_points(2, 4, 5.5, 5.6)
    else:
        raise ValueError('Invalid mode: '.format(mode))
elif dataset_type == 'extrapolation':
    theta_min1, theta_max1 = np.pi/2, np.pi*5/6
    theta_min2, theta_max2 = np.pi*7/6, np.pi*3/2
    if mode == 'training':
        dataset_name = 'training_extrapolation'
        n_theta = 5
        d = (theta_max1 - theta_min1) * r / n_theta
        points1 = cylinder_points(r, n_theta, 0.5, 6.5, theta_min=theta_min1, 
            theta_max=theta_max1, add_last=True)
        points1_in = cylinder_points(r-d, n_theta, 0.5, 6.5, theta_min=theta_min1, 
            theta_max=theta_max1, add_last=True, d=d)
        points1_out = cylinder_points(r+d, n_theta, 0.5, 6.5, theta_min=theta_min1, 
            theta_max=theta_max1, add_last=True, d=d)
        points2 = cylinder_points(r, n_theta, 0.5, 6.5, theta_min=theta_min2, 
            theta_max=theta_max2, add_last=True)
        points2_in = cylinder_points(r-d, n_theta, 0.5, 6.5, theta_min=theta_min2, 
            theta_max=theta_max2, add_last=True, d=d)
        points2_out = cylinder_points(r+d, n_theta, 0.5, 6.5, theta_min=theta_min2, 
            theta_max=theta_max2, add_last=True, d=d)
        points = np.concatenate([points1_in, points2_in, points1, points2, 
            points1_out, points2_out], axis=0)
    elif mode == 'validation':
        dataset_name = 'validation_extrapolation'
        points1 = random_cylinder_points(45, 2.5, 4.5, 0.5, 6.5, 
            theta_min=theta_min1, theta_max=theta_max1)
        points2 = random_cylinder_points(45, 2.5, 4.5, 0.5, 6.5, 
            theta_min=theta_min2, theta_max=theta_max2)
        points = np.concatenate([points1, points2], axis=0)
    elif mode == 'test1':
        dataset_name = 'test1_extrapolation'
        points = []
else:
    raise ValueError('Invalid dataset type: '.format(dataset_type))

n_images = points.shape[0]
print('Points generated:', n_images)
time.sleep(2)

#import matplotlib
#from mpl_toolkits.mplot3d import Axes3D
#import matplotlib.pyplot as plt

#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#ax.plot(points[:,0], points[:,1], points[:,2], lw=0, marker='.')
#ax.set_xlim(-2,2)
#ax.set_ylim(-2,2)
#ax.set_zlim(-2,2)
#plt.show()

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

with open(os.path.join(output_dir, dataset_name + '.txt'), 'w') as f:
    # Write header
    f.write('{}\n'.format(dataset_name))
    f.write('ImageFile, Camera Position [X Y Z W P Q R]\n\n')

    fnumber_format = int(math.ceil(math.log10(n_images)))
    camera = Camera(*size)

    for i in range(n_images):
        # Set the camera location and orientation
        x, y, z = points[i,:]
        camera.setLocation((x, y, z))
        camera.look_at([random.uniform(-dv, dv), random.uniform(-dv, dv), z])

        fnumber = str(i).zfill(fnumber_format)
        fname = os.path.join(dataset_name, 'image{}.png'.format(fnumber))

        # Render current view and save it
        camera.takePicture(os.path.join(output_dir, fname))

        # Write the camera pose to a file
        f.write('{} {}\n'.format(fname, camera.getPoseString()))

    print('\nGenerated {} images'.format(n_images))