import argparse
import math
import os
import sys

from posenet.core.image_reader import ImageReader, read_label_file
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

input_size = 224
test_reader = ImageReader(args.dataset, batch_size=1, 
                    image_size=[input_size, input_size], 
                    random_crop=False, randomise=False)
n_images = test_reader.total_images()

# Read the definition file to get file names
paths, _ = read_label_file(args.dataset)
paths = map(lambda x: os.path.basename(x), paths)

# Localise
with Localiser(input_size, args.model, uncertainty=args.uncertainty) as localiser:
    if args.output:
        f = open(args.output, 'w')
        f.write('\n\n\n') # Hack for now

    for i in range(n_images):
        images_feed, labels_feed = test_reader.next_batch()

        # Make prediction
        predicted = localiser.localise(images_feed)
        x = [round(v, 6) for v in predicted['x']]
        q = [round(v, 6) for v in predicted['q']]

        if args.output:
            f.write('{} {} {} {} {} {} {} {}\n'.format(paths[i], x[0], x[1], x[2], q[0], q[1], q[2], q[3]))
            progress_bar(1.0*(i+1)/n_images, 30, text='Localising')
        else:
            std_x = round(predicted['std_x'], 6)
            std_q = round(predicted['std_q'], 6)
            print('---- {} ----'.format(paths[i]))
            print('x: {}'.format(x))
            print('q: {}'.format(q))
            print('std_x: {}'.format(std_x))
            print('std_q: {}'.format(std_q))
    print('')

    if args.output:
        f.close()