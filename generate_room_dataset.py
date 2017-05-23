import os
import random
import sys
import time

sys.path.append(os.path.dirname(__file__)) # So that next imports work
from posenet.blender import *
from posenet.utils import *


output_dir = 'datasets/room'
mode = 'test2'

vol_x = (-6, 6)
vol_y = (-6, 6)
vol_z = (2, 6)
r = 100
phi = (-3.14159, 3.14159)
theta = (1.05, 2.09)
size = (256, 256)

if mode == 'training':
    dataset_name = 'training'
    n_orientations = 500
    factor = 4
    ox, oy, oz = sample_spherical(n_orientations, phi1=phi[0], phi2=phi[1],
                            theta1=theta[0], theta2=theta[1])
elif mode == 'validation':
    dataset_name = 'validation'
    n_orientations = 400
    factor = 1
    ox, oy, oz = sample_spherical(n_orientations, phi1=phi[0], phi2=phi[1],
                            theta1=theta[0], theta2=theta[1])
elif mode == 'test1':
    dataset_name = 'test1'
    vol_x = (0.0, 0.0)
    vol_y = (0.0, 0.0)
    vol_z = (3.5, 3.5)
    factor = 1
    n_orientations = 180
    points = cylinder_points(r, n_orientations, 4, 4)
    ox, oy, oz = points[:,0], points[:,1], points[:,2]
elif mode == 'test2':
    dataset_name = 'test2'
    n_orientations = 400
    factor = 1
    ox, oy, oz = sample_spherical(n_orientations, phi1=phi[0], phi2=phi[1],
                            theta1=theta[0], theta2=theta[1])
else:
    raise ValueError('Invalide dataset_name')

if not os.path.exists(output_dir):
    os.makedirs(output_dir)


n_images = factor * len(ox)
print('Points generated:', n_images)
time.sleep(2)

with open(os.path.join(output_dir, dataset_name + '.txt'), 'w') as f:
    # Write header
    f.write('{}\n'.format(dataset_name))
    f.write('ImageFile, Camera Position [X Y Z W P Q R]\n\n')

    fnumber_format = int(math.ceil(math.log10(n_images)))
    camera = Camera(size[0], size[1])

    img_num = 0
    for i in range(len(ox)):
        oxi, oyi, ozi = r*ox[i], r*oy[i], r*oz[i]
        for j in range(factor):
            x = random.uniform(*vol_x)
            y = random.uniform(*vol_y)
            z = random.uniform(*vol_z)

            camera.setLocation((x, y, z))
            camera.look_at((oxi, oyi, ozi))

            fnumber = str(img_num).zfill(fnumber_format)
            fname = os.path.join(dataset_name, 'image{}.png'.format(fnumber))

            # Render current view and save it
            camera.takePicture(os.path.join(output_dir, fname))

            # Write the camera pose to a file
            f.write('{} {}\n'.format(fname, camera.getPoseString()))
            img_num += 1