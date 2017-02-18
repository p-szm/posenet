import argparse
import glob
import math
import os
import sys

from posenet.core.image_reader import read_image, read_label_file
from posenet.core.localiser import Localiser
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
elif args.dataset.endswith('.txt'):
    imgs, _ = read_label_file(args.dataset, full_paths=True)
elif args.dataset.endswith('.png'):
    imgs = [args.dataset]
else:
    imgs = []

n_images = len(imgs)
if n_images == 0:
    print('No images found')
    sys.exit(1)

# Localise
input_size = 256
with Localiser(args.model, uncertainty=args.uncertainty) as localiser:
    if args.output:
        f = open(args.output, 'w')
        f.write('\n\n\n') # Hack for now

    for i in range(n_images):
        # Load image
        image = read_image(imgs[i], normalise=True, size=[input_size, input_size])

        # Make prediction
        predicted = localiser.localise(image)
        x = [round(v, 6) for v in predicted['x']]
        q = [round(v, 6) for v in predicted['q']]

        fname = os.path.basename(imgs[i])
        if args.output:
            f.write('{} {} {} {} {} {} {} {}\n'.format(
                fname, x[0], x[1], x[2], q[0], q[1], q[2], q[3]))
            progress_bar(1.0*(i+1)/n_images, 30, text='Localising')
        else:
            print('---- {} ----'.format(fname))
            print('x: {}'.format(x))
            print('q: {}'.format(q))
            if args.uncertainty:
                std_x = round(predicted['std_x'], 6)
                std_q = round(predicted['std_q'], 6)
                print('std_x: {}'.format(std_x))
                print('std_q: {}'.format(std_q))
    print('')

    if args.output:
        f.close()