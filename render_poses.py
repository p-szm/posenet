import argparse
import os
import sys

import numpy as np

sys.path.append(os.path.dirname(__file__)) # So that next imports work
from posenet.blender import *
from posenet.utils import *

parser = argparse.ArgumentParser()
parser.add_argument('-p', '--pose', type=float, nargs='*', action='store', required=False)
parser.add_argument('-f', '--def_file', action='store', required=False)
parser.add_argument('-o', '--output', type=str, action='store', required=True)
parser.add_argument('-s', '--size', type=int, nargs=2, action='store', required=False, 
                    default=[300,300])
args = parser.parse_args(preprocess_args(sys.argv))


def read_label_file(def_file):
    paths = []
    labels = []
    with open(def_file) as f:
        lines = list(map(lambda line: line.rstrip("\n").split(" "), f.readlines()[3:]))
        paths = list(map(lambda line: line[0], lines))
        labels = list(map(lambda line: list(map(lambda x: float(x), line[1:])), lines))
    return list(paths), list(labels)

def split_pose(pose):
    x = np.array(pose[0:3])
    q = np.array(pose[3:])
    q = q/np.linalg.norm(q)
    return x, q


if not args.def_file and not args.pose:
    print('Definition file or pose required')
    sys.exit(1)

if args.pose and len(args.pose) < 6:
    print('Invalide pose')
    sys.exit(1)

camera = Camera(width=args.size[0], height=args.size[1])

if args.def_file:
    paths, labels = read_label_file(args.def_file)
    for path, label in zip(paths, labels):
        x, q = split_pose(label)
        camera.setLocation(x)
        if len(q) == 3:
            camera.setAxis(q)
        else:
            camera.setRotation(q)
        camera.takePicture(os.path.join(args.output, path))
elif args.pose:
    x, q = split_pose(args.pose)
    camera.setLocation(x)
    if len(q) == 3:
        camera.setAxis(q)
    else:
        camera.setRotation(q)
    camera.takePicture(args.output)
