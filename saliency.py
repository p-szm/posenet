import argparse
import glob
import os
import sys

import numpy as np
import scipy
from skimage import io, transform
from skimage.exposure import adjust_gamma

from posenet.core.localiser import Localiser
from posenet.core.image_reader import read_image, read_label_file
from posenet.utils import progress_bar


parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', action='store', required=True, 
    help='''Path to a trained Tensorflow model (.ckpt file)''')
parser.add_argument('-d', '--dataset', action='store', required=True, 
    help='''Path to a text file listing images and camera poses''')
parser.add_argument('-o', '--output', action='store', required=False)
args = parser.parse_args()    


if os.path.isdir(args.dataset):
    imgs = glob.glob('{}/*.png'.format(args.dataset))
elif args.dataset.endswith('.txt'):
    imgs, _ = read_label_file(args.dataset, full_paths=True)
elif args.dataset.endswith('.png'):
    imgs = [args.dataset]
    from matplotlib import pyplot as plt
else:
    imgs = []


n_images = len(imgs)
if n_images > 1 and not args.output:
    print('--output argument required')
    sys.exit(1)
if n_images == 0:
    print('No images found')
    sys.exit(1)

if args.output and not os.path.isdir(args.output):
    os.makedirs(args.output)

# Localise
input_size = 224
with Localiser(input_size, args.model) as localiser:
    for i in range(n_images):
        # Load normalised image
        image = read_image(imgs[i], normalise=True, size=[input_size, input_size])

        # Compute the saliency map
        grad = localiser.saliency(image)
        grad = adjust_gamma(grad, gamma=0.8)

        if args.output:
            fname = os.path.join(args.output, os.path.basename(imgs[i]))
            scipy.misc.imsave(fname, grad)
            progress_bar(1.0*(i+1)/n_images, 30, text='Computing')
        else:
            # Display image
            plt.imshow(grad, cmap='gray')
            plt.show()

    if n_images > 1:
        print('')
