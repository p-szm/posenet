from image_reader import ImageReader
from localiser import Localiser

input_size = 224
test_reader = ImageReader('datasets/monkey/test2.txt', 1, [input_size, input_size], False, False)

with Localiser(input_size, 'models/model14.ckpt') as localiser:
	for i in xrange(test_reader.total_images()):
		images_feed, labels_feed = test_reader.next_batch()
		predicted = localiser.localise(images_feed)
		print predicted