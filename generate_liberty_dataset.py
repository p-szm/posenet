import math
import os
import random
import sys
import time
import numpy as np

sys.path.append(os.path.dirname(__file__)) # So that next imports work
from posenet.blender import *
from posenet.utils import *


output_dir = 'datasets/liberty'
mode = 'test1'
size = [512, 512]
z_min, z_max = 0, 5.5

if mode == 'training':
    dataset_name = 'training'
    points1 = cylinder_points(1.5, 20, z_min, z_max)
    points2 = cylinder_points(2, 25, z_min, z_max)
    points3 = cylinder_points(2.5, 30, z_min, z_max)
    points4 = cylinder_points(3, 40, z_min, z_max)
    points5 = cylinder_points(3.5, 45, z_min, z_max)
    points6 = cylinder_points(4, 35, z_min, z_max)
    points7 = cylinder_points(4.5, 25, z_min, z_max)
    points = np.concatenate([points1, points2, points3, 
        points4, points5, points6, points7], axis=0)
    dv = 0.5
elif mode == 'validation':
    dataset_name = 'validation'
    points = random_cylinder_points(300, 1.2, 4.8, z_min, z_max)
    dv = 0.5
elif mode == 'test1':
    dataset_name = 'test1'
    points = cylinder_points(3, 200, 3.9, 3.9)
    dv = 0

n_images = points.shape[0]
print('Points generated:', n_images)
time.sleep(2)

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

with open(os.path.join(output_dir, dataset_name + '.txt'), 'w') as f:
    # Write header
    f.write('{}\n'.format(dataset_name))
    f.write('ImageFile, Camera Position [X Y Z W P Q R]\n\n')

    fnumber_format = 4# int(math.ceil(math.log10(n_images)))
    camera = Camera(*size)

    for i in range(n_images):
        x, y, z = points[i,:]
        camera.setLocation((x, y, z))
        camera.look_at([random.uniform(-dv, dv), random.uniform(-dv, dv), z])

        fnumber = str(i).zfill(fnumber_format)
        fname = os.path.join(dataset_name, 'image{}.png'.format(fnumber))

        camera.takePicture(os.path.join(output_dir, fname))
        f.write('{} {}\n'.format(fname, camera.getPoseString()))

    print('\nGenerated {} images'.format(n_images))