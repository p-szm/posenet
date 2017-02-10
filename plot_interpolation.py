import argparse

import matplotlib
import numpy as np

from posenet.core.image_reader import ImageReader
from posenet.core.localiser import Localiser
from posenet.utils import progress_bar
from posenet.utils import to_spherical

parser = argparse.ArgumentParser()
parser.add_argument('--agg', action='store_true')
parser.add_argument('--model', action='store', required=True)
parser.add_argument('--dataset', action='store', required=True)
parser.add_argument('--limits', action='store', nargs=2, type=int, 
                    required=False, default=[-90,90])
parser.add_argument('--spacing', action='store', type=int, required=True)
parser.add_argument('--verbose', action='store_true')
parser.add_argument('--save', action='store', required=False)
args = parser.parse_args()

if args.agg:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

input_size = 224
test_reader = ImageReader(args.dataset, batch_size=1, 
                        image_size=[input_size, input_size],
                        randomise=False)
n_images = test_reader.total_images()

azimuthal = []
with Localiser(args.model) as localiser:
    for i in range(n_images):
        images_feed, labels_feed = test_reader.next_batch()
        gt = {'x': labels_feed[0][0:3], 'q': labels_feed[0][3:7]}

        # Make prediction
        predicted = localiser.localise(images_feed)
        x, y, z = predicted['x']
        azimuthal.append(to_spherical(x, y, z)[1]*180/np.pi)

        if args.verbose:
            print('-------------{}-------------'.format(i))
            print(predicted)
        else:
            progress_bar(1.0*(i+1)/n_images, 30, text='Localising')
    print('')


fig = plt.figure()
fig.patch.set_facecolor('white')
x = np.linspace(args.limits[0], args.limits[1], n_images)
plt.plot(x, azimuthal, color='black')
plt.plot(x[0::args.spacing], azimuthal[0::args.spacing], 'ro', ms=4)
plt.xlim([args.limits[0], args.limits[1]])
plt.ylabel("Predicted azimuthal angle")
plt.xlabel("True azimuthal angle")

if args.save:
    plt.savefig(args.save, bbox_inches='tight')
else:
    plt.show()