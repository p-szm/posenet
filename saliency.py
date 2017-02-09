import argparse
import glob
import os
import sys

import numpy as np
import scipy
from skimage import io, transform
from skimage.exposure import adjust_gamma

from posenet.core.localiser import Localiser
from posenet.core.image_reader import read_label_file
from posenet.utils import progress_bar


parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', action='store', required=True, 
    help='''Path to a trained Tensorflow model (.ckpt file)''')
parser.add_argument('-d', '--dataset', action='store', required=True, 
    help='''Path to a text file listing images and camera poses''')
parser.add_argument('-o', '--output', action='store', required=False)
parser.add_argument('-u', '--uncertainty', action='store_true')
args = parser.parse_args()    


if os.path.isdir(args.dataset):
    imgs = glob.glob('{}/*.png'.format(args.dataset))
elif '.txt' in args.dataset:
    imgs, _ = read_label_file(args.dataset)
    imgs = map(lambda x: os.path.join(os.path.dirname(args.dataset), x), imgs)
elif '.png' in args.dataset:
    imgs = [args.dataset]
    from matplotlib import pyplot as plt

n_images = len(imgs)
if n_images > 1 and not args.output:
    print('--output argument required')
    sys.exit(1)
if n_images == 0:
    print('No images found')
    sys.exit(1)

if args.output and not os.path.isdir(args.output):
    os.makedirs(args.output)

input_size = 224

# Localise
with Localiser(input_size, args.model, uncertainty=args.uncertainty) as localiser:
    for i in range(n_images):
        # Load and preprocess image
        image = io.imread(imgs[i])
        image = transform.resize(image, (224, 224, 3))
        image_input = np.expand_dims(2.0*image - 1.0, axis=0)

        # Compute the saliency map
        grad = localiser.saliency(image_input)
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
