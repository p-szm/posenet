import argparse
import math
import os
import random
import sys

import bpy
from mathutils import *

sys.path.append(os.path.dirname(bpy.data.filepath)) # So that next import works
from utils import *

"""Renders views of an object from different sides"""


parser = argparse.ArgumentParser()
parser.add_argument('--output_dir', action='store', required=True)
parser.add_argument('--dataset_name', action='store', required=True)
parser.add_argument('--width', action='store', type=int, default=1000)
parser.add_argument('--height', action='store', type=int, default=800)
parser.add_argument('--n_images', action='store', type=int, default=10)
parser.add_argument('--origin', action='store', type=float, nargs=3, default=[0,0,0])
parser.add_argument('--vary_origin', action='store', type=float, default=0)
parser.add_argument('--r', action='store', default='10')
parser.add_argument('--phi', action='store', default='uniform[-3.1416,3.1416]')
parser.add_argument('--theta', action='store', default='uniform[-1.5708,1.5708]')
args = parser.parse_args(preprocess_args(sys.argv))


scene = bpy.data.scenes["Scene"]
camera = Camera(scene.camera)

r_interval = parse_interval(args.r)
phi_interval = parse_interval(args.phi)
theta_interval = parse_interval(args.theta)

def sample_from_interval(interval, i):
    if type(interval) is int or type(interval) is float:
        # Constant
        return interval
    if len(interval) == 3:
        interval_type = interval[2]
        if interval_type == 'uniform':
            return random.uniform(interval[0], interval[1])
        elif interval_type == 'linspace':
            return interval[0] + i * (interval[1] - interval[0]) / (args.n_images - 1)
        else:
            raise ValueError('Unknown interval type: {}'.format(interval_type))

if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

with open(os.path.join(args.output_dir, args.dataset_name + '.txt'), 'w') as f:
    # Write header
    f.write('{}\n'.format(args.dataset_name))
    f.write('ImageFile, Camera Position [X Y Z W P Q R]\n\n')

    fnumber_format = int(math.ceil(math.log10(args.n_images)))

    for i in range(args.n_images):
        r = sample_from_interval(r_interval, i)
        phi = sample_from_interval(phi_interval, i)
        theta = sample_from_interval(theta_interval, i)
        x = args.origin[0] + r * math.sin(theta) * math.cos(phi)
        y = args.origin[1] + r * math.sin(theta) * math.sin(phi)
        z = args.origin[2] + r * math.cos(theta)

        # Set the camera location and orientation
        camera.setLocation(Vector((x, y, z)))
        var = args.vary_origin
        camera.look_at(Vector((args.origin[0] + random.uniform(-var, var), 
                               args.origin[1] + random.uniform(-var, var), 
                               args.origin[2] + random.uniform(-var, var))))
        q = camera.getRotation()

        fnumber = str(i).zfill(fnumber_format)
        fname = os.path.join(args.dataset_name, 'image{}.png'.format(fnumber))

        # Render current view and save it
        renderToFile(os.path.join(args.output_dir, fname), args.width, args.height)

        # Write the camera pose to a file
        f.write('{} {}\n'.format(fname, camera.getPoseString()))
