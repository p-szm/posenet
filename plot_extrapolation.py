import argparse

import matplotlib
import numpy as np

from posenet.core.image_reader import ImageReader
from posenet.core.localiser import Localiser
from posenet.utils import l2_distance, quaternion_distance, progress_bar

parser = argparse.ArgumentParser(description='''
    Tests a model that was trained on a single object from multiple sides''')
parser.add_argument('--agg', action='store_true')
parser.add_argument('--model', action='store', required=True, 
    help='''Path to a trained Tensorflow model (.ckpt file)''')
parser.add_argument('--dataset', action='store', required=True, 
    help='''Path to a text file listing images and camera poses''')
parser.add_argument('--test_limits', action='store', nargs=2, type=int, 
                    required=False, default=[-90,90])
parser.add_argument('--training_limits', action='store', nargs=2, type=int, 
                    required=False, default=[-45,45])
parser.add_argument('--verbose', action='store_true')
parser.add_argument('--save', action='store', required=False)
args = parser.parse_args()

if args.agg:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

input_size = 224
test_reader = ImageReader(args.dataset, batch_size=1, 
                        image_size=[input_size, input_size], 
                        random_crop=False, randomise=False)
n_images = test_reader.total_images()

pos_errors = []
orient_errors = []
positions = np.empty([0,3])
orientations = np.empty([0,4])

with Localiser(input_size, args.model) as localiser:
    for i in range(n_images):
        images_feed, labels_feed = test_reader.next_batch()
        gt = {'x': labels_feed[0][0:3], 'q': labels_feed[0][3:7]}

        # Make prediction
        predicted = localiser.localise(images_feed)
        x, y, z = predicted['x']
        q1, q2, q3, q4 = predicted['q']

        pos_error = l2_distance(gt['x'], predicted['x'])
        orient_error = quaternion_distance(gt['q'], predicted['q']) * 180 / np.pi
        pos_errors.append(pos_error)
        orient_errors.append(orient_error)

        if args.verbose:
            print('-------------{}-------------'.format(i))
            print(predicted)
            print('Positional error:  {}'.format(pos_error))
            print('Orientation error: {}'.format(orient_error))
        else:
            progress_bar(1.0*(i+1)/n_images, width=30, text='Localising')
    print('')


x = np.linspace(args.test_limits[0], args.test_limits[1], n_images)

fig, axes = plt.subplots(nrows=2, ncols=1)
fig.patch.set_facecolor('white')
ax1 = axes[0]
ax2 = axes[1]

ax1.plot(x, pos_errors, color='black')
ax1.set_xlim([args.test_limits[0], args.test_limits[1]])
shade = (1, 0.8, 0.8)
ax1.axvspan(args.test_limits[0], args.training_limits[0], color=shade)
ax1.axvspan(args.training_limits[1], args.test_limits[1], color=shade)
ax1.set_ylabel("Positional error")

ax2.plot(x, orient_errors, color='black')
ax2.set_xlim([args.test_limits[0], args.test_limits[1]])
ax2.axvspan(args.test_limits[0], args.training_limits[0], color=shade)
ax2.axvspan(args.training_limits[1], args.test_limits[1], color=shade)
ax2.set_xlabel('Phi')
ax2.set_ylabel("Orientation error (degrees)")

if args.save:
    plt.savefig(args.save, bbox_inches='tight')
else:
    plt.show()