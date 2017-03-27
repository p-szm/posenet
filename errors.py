import argparse
import glob
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from posenet.core.image_reader import *
from posenet.core.localiser import Localiser
from posenet.utils import progress_bar

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', '-d', action='store')
parser.add_argument('--model_dir', '-m', action='store')
parser.add_argument('--samples', '-s', action='store', type=int, default=40)
parser.add_argument('--dropout', action='store', type=float, default=0.5)
parser.add_argument('--input_size', action='store', nargs=2, type=int, default=[256,256])
parser.add_argument('--last_iter', action='store', type=int, default=100000)
args = parser.parse_args()


def localise_all(model, img_paths):
    x, q = [], []
    std_x, std_q = [], []
    with Localiser(model, uncertainty=True, output_type='axis',
               dropout=args.dropout) as localiser:
        for i, path in enumerate(img_paths):
            img = read_image(path, size=args.input_size, 
                    expand_dims=False, normalise=True)
            pred = localiser.localise(img, samples=args.samples)
            x.append(pred['x'])
            q.append(pred['q'])
            std_x.append(pred['std_x_all'])
            std_q.append(pred['std_q_all'])
    return np.array(x), np.array(q), np.array(std_x), np.array(std_q)

def model_list(model_dir):
    models = glob.glob(os.path.join(model_dir, '*.ckpt.data*'))
    models = [m.split('.data-00000-of-00001')[0] for m in models]
    return models

def iterations(models):
    # Sort using the iteration number
    iters = [m[:-5].split('iter')[-1] if ('iter' in m) else None for m in models]
    iters = [int(i) if i is not None else None for i in iters]
    if None in iters:
        iters[iters.index(None)] = args.last_iter
    return iters

def rearrange(iters, models):
    it_model = sorted(zip(iters, models))
    return zip(*it_model)


img_paths, labels = read_label_file(args.dataset, full_paths=True, 
                    convert_to_axis=True)
n_images = len(img_paths)
x_gt = np.array([l[0:3] for l in labels])
q_gt = np.array([l[3:] for l in labels])

if os.path.isdir(args.model_dir):
    models = model_list(args.model_dir)
else:
    models = [args.model_dir]

iters = iterations(models)
iters, models = rearrange(iters, models)

for model in models:
    x, q, std_x, std_q = localise_all(model, img_paths)
    l2x = np.mean(np.linalg.norm(x-x_gt, ord=2, axis=1))
    l2q = np.mean(np.linalg.norm(q-q_gt, ord=2, axis=1))
    print os.path.basename(model), l2x, l2q