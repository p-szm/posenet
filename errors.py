import argparse
import glob
import os
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', '-d', action='store')
parser.add_argument('--model_dir', '-m', action='store')
parser.add_argument('--samples', '-s', action='store', type=int, default=40)
parser.add_argument('--dropout', action='store', type=float, default=0.5)
parser.add_argument('--size', action='store', nargs=2, type=int, default=[256,256])
parser.add_argument('--save', action='store', required=True)
parser.add_argument('--agg', action='store_true')
args = parser.parse_args()

import matplotlib
if args.agg:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from posenet.core.image_reader import *
from posenet.core.localiser import Localiser
from posenet.utils import progress_bar


def localise_all(model, img_paths):
    x, q = [], []
    std_x, std_q = [], []
    with Localiser(model, uncertainty=False, output_type='axis',
               dropout=args.dropout) as localiser:
        for i, path in enumerate(img_paths):
            img = read_image(path, expand_dims=False, normalise=True)
            img = resize_image(centre_crop(img), args.size)
            pred = localiser.localise(img, samples=args.samples)
            x.append(pred['x'])
            q.append(pred['q'])
            #std_x.append(pred['std_x_all'])
            #std_q.append(pred['std_q_all'])
    return np.array(x), np.array(q)#, np.array(std_x), np.array(std_q)

def model_list(model_dir):
    models = glob.glob(os.path.join(model_dir, '*_iter*.ckpt.data*'))
    models = [m.split('.data-00000-of-00001')[0] for m in models]
    return models

def iterations(models):
    # Sort using the iteration number
    iters = [m[:-5].split('iter')[-1] if ('iter' in m) else None for m in models]
    iters = [int(i) for i in iters if i is not None]
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

x_errors = []
q_errors = []
for model in models:
    x, q = localise_all(model, img_paths)
    x_errors.append(np.mean(np.linalg.norm(x-x_gt, ord=2, axis=1)))
    q_errors.append(180.0/np.pi*np.mean(np.arccos(np.sum(q*q_gt, axis=1))))
print iters, x_errors, q_errors


ax1 = plt.subplot(211)
plt.plot(iters, x_errors)
plt.xlim([0, None])
plt.ylim([0, None])
plt.ylabel('Position error')

ax2 = plt.subplot(212, sharex=ax1)
plt.plot(iters, q_errors)
plt.xlim([0, None])
plt.ylim([0, None])
plt.ylabel('Orientation error (degrees)')
plt.xlabel('Iteration')

if not os.path.isdir(os.path.dirname(args.save)):
    os.makedirs(os.path.dirname(args.save))

plt.savefig(args.save, bbox_inches='tight')
