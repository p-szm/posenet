import argparse

import numpy as np

from image_reader import ImageReader
from localiser import Localiser
from utils import l2_distance, quaternion_distance


parser = argparse.ArgumentParser()
parser.add_argument('--model', action='store', required=True)
parser.add_argument('--dataset', action='store', required=True)
parser.add_argument('--plot_errors', action='store', nargs='?', const='')
args = parser.parse_args()


input_size = 224
test_reader = ImageReader(args.dataset, batch_size=1, 
						image_size=[input_size, input_size], 
						random_crop=False, randomise=False)

pos_errors = []
orient_errors = []

with Localiser(input_size, args.model) as localiser:
	print 'Localising...'
	for i in xrange(test_reader.total_images()):
		images_feed, labels_feed = test_reader.next_batch()
		gt = {'x': labels_feed[0][0:3], 'q': labels_feed[0][3:7]}

		# Make prediction
		predicted = localiser.localise(images_feed)

		pos_error = l2_distance(gt['x'], predicted['x'])
		orient_error = quaternion_distance(gt['q'], predicted['q']) * 180 / np.pi
		r = l2_distance(predicted['x'], [0,0,0])

		pos_errors.append(pos_error)
		orient_errors.append(orient_error)

		print('-------------{}-------------'.format(i))
		print('Positional error:  {}'.format(pos_error))
		print('Orientation error: {}'.format(orient_error))
	print 'Done'


if args.plot_errors is not None:
	import matplotlib.pyplot as plt

	x = np.linspace(-90, 90, test_reader.total_images())

	fig, axes = plt.subplots(nrows=2, ncols=1)
	fig.patch.set_facecolor('white')
	ax1 = axes[0]
	ax2 = axes[1]

	ax1.plot(x, pos_errors, color='black')
	ax1.set_xlim([-90, 90])
	ax1.axvspan(-90, -45, alpha=0.2, color='red')
	ax1.axvspan(45, 90, alpha=0.2, color='red')
	ax1.set_ylabel("Positional error")

	ax2.plot(x, orient_errors, color='black')
	ax2.set_xlim([-90, 90])
	ax2.axvspan(-90, -45, alpha=0.2, color='red')
	ax2.axvspan(45, 90, alpha=0.2, color='red')
	ax2.set_xlabel('Phi')
	ax2.set_ylabel("Orientation error (degrees)")

	if args.plot_errors:
		plt.savefig(args.plot_errors)
	else:
		plt.show()