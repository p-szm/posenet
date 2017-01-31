import argparse
import os
import sys

import numpy as np
import skimage

sys.path.append(os.path.dirname(__file__)) # So that next imports work
from posenet.blender import *
from posenet.utils import *


parser = argparse.ArgumentParser(description='''
    Renders a single frame of a .blend file.
    Usage: blender -b -P from_different_sides.py -- [...arguments...]''')
parser.add_argument('--pose', type=float, nargs=7, action='store', required=True)
parser.add_argument('--output', type=str, action='store', required=True)
parser.add_argument('--size', type=int, nargs=2, action='store', required=False, 
					default=[300,300])
args = parser.parse_args(preprocess_args(sys.argv))

x = np.array(args.pose[0:3])
q = np.array(args.pose[3:7])
q = q/np.linalg.norm(q)

camera = Camera(width=args.size[0], height=args.size[1])
camera.setLocation(x)
camera.setRotation(q)
camera.takePicture(args.output)

# To blend images:
# convert image00.png image01.png -evaluate-sequence mean result.png