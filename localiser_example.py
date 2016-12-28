import argparse

from image_reader import ImageReader
from localiser import Localiser
from utils import l2_distance, quaternion_distance


parser = argparse.ArgumentParser()
parser.add_argument('--model', action='store', required=True)
parser.add_argument('--dataset', action='store', required=True)
args = parser.parse_args()


input_size = 224
test_reader = ImageReader(args.dataset, 1, [input_size, input_size], False, False)

with Localiser(input_size, args.model) as localiser:
	for i in xrange(test_reader.total_images()):
		images_feed, labels_feed = test_reader.next_batch()
		gt = {'x': labels_feed[0][0:3], 'q': labels_feed[0][3:7]}

		# Make prediction
		predicted = localiser.localise(images_feed)

		pos_error = l2_distance(gt['x'], predicted['x'])
		orient_error = quaternion_distance(gt['q'], predicted['q'])

		print('-------------{}-------------'.format(i))
		print('Positional error:  {}'.format(pos_error))
		print('Orientation error: {}'.format(orient_error))
