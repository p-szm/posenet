import argparse
import math
import os
import random
import sys

import numpy as np

sys.path.append(os.path.dirname(__file__)) # So that next imports work
from posenet.blender import *
from posenet.utils import *


parser = argparse.ArgumentParser(description='''
    Renders images of an object from multiple sides. For use with a .blend file. 
    Usage: blender -b -P from_different_sides.py -- [...arguments...]''',
    epilog='''Intervals from a to b: uniform[a,b] or linspace[a,b]''')
parser.add_argument('--output_dir', action='store', required=True,
    help='''Directory to which the folder with images and definition file 
    will be saved''')
parser.add_argument('--dataset_name', action='store', required=True,
    help='''Name for the folder with images and definition file which will 
    be produced (a .txt extension will be added to the definition file)''')
parser.add_argument('--width', action='store', type=int, default=500,
    help='''Width of the rendered images''')
parser.add_argument('--height', action='store', type=int, default=400,
    help='''Height of the rendered images''')
parser.add_argument('--n_images', action='store', type=int, default=1,
    help='''Number of images to be produced''')
parser.add_argument('--origin', action='store', type=float, nargs=3, default=[0,0,0],
    help='''Coordinates of the point on which the camera will look''')
parser.add_argument('--vary_origin', action='store', type=float, default=0,
    help='''The amount by which the origin point will be moved in each 
    direction randomly''')
parser.add_argument('--spherical', action='store', type=float, nargs=4,
    help='''Camera poses are distributed uniformly on a sphere part defined by 
    [phi_start, phi_end, theta_start, theta_end]''')
parser.add_argument('--cap', action='store', type=float, nargs=3,
    help='''Camera poses are distributed uniformly on a sphere cap centred at 
    [phi, theta]
    and spanning an angle 2*alpha. Arguments: [phi,theta,alpha]''')
parser.add_argument('--linear', action='store', type=float, nargs=4,
    help='''Camera poses move linearly on a path defined by 
    [phi_start, phi_end, theta_start, theta_end]''')
parser.add_argument('--r', action='store', type=float, nargs=2, default=[8,8],
    help='''Radius varies randomly (uniform) in the range [r_star, r_end]''')
parser.add_argument('--factor', action='store', type=int, default=1)
args = parser.parse_args(preprocess_args(sys.argv))

def repeat(lst, k):
    return [v for v in lst for i in range(k)]

if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

# Generate camera poses
n_poses = int(math.ceil(1.0*args.n_images/args.factor))
if args.spherical:
    x, y, z = sample_spherical(n_poses, 
                               phi1=args.spherical[0], phi2=args.spherical[1],
                               theta1=args.spherical[2], theta2=args.spherical[3])
elif args.cap:
    x, y, z = sample_cap(n_poses, cap_phi=args.cap[0], cap_theta=args.cap[1],
                         cap_alpha=args.cap[2])
elif args.linear:
    phi = np.linspace(args.linear[0], args.linear[1], args.n_images)
    theta = np.linspace(args.linear[2], args.linear[3], args.n_images)
    x, y, z = to_cartesian(phi, theta)

x = repeat(x, args.factor)
y = repeat(y, args.factor)
z = repeat(z, args.factor)

n_images = len(x)

with open(os.path.join(args.output_dir, args.dataset_name + '.txt'), 'w') as f:
    # Write header
    f.write('{}\n'.format(args.dataset_name))
    f.write('ImageFile, Camera Position [X Y Z W P Q R]\n\n')

    fnumber_format = int(math.ceil(math.log10(n_images)))

    camera = Camera(args.width, args.height)

    for i in range(n_images):
        r = random.uniform(args.r[0], args.r[1])

        # Set the camera location and orientation
        camera.setLocation(Vector((args.origin[0] + r*x[i], 
                                   args.origin[1] + r*y[i], 
                                   args.origin[2] + r*z[i])))
        var = args.vary_origin
        camera.look_at(Vector((args.origin[0] + random.uniform(-var, var), 
                               args.origin[1] + random.uniform(-var, var), 
                               args.origin[2] + random.uniform(-var, var))))

        fnumber = str(i).zfill(fnumber_format)
        fname = os.path.join(args.dataset_name, 'image{}.png'.format(fnumber))

        # Render current view and save it
        camera.takePicture(os.path.join(args.output_dir, fname))

        # Write the camera pose to a file
        f.write('{} {}\n'.format(fname, camera.getPoseString()))

        print(i, file=sys.stderr)

    print('\nGenerated {} images'.format(n_images))
