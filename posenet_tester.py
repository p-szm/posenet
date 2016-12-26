import tensorflow as tf
import tensorflow.contrib.slim as slim
import argparse
import os
import sys
from posenet import Posenet
from image_reader import ImageReader

parser = argparse.ArgumentParser()
parser.add_argument('--model', action='store', required=True)
parser.add_argument('--dataset', action='store', required=True)
parser.add_argument('--save_output', action='store')
args = parser.parse_args()

if not os.path.isfile(args.model) or os.path.isdir(args.model):
	print '{} does not exist or is a directory'.format(args.model)
	sys.exit()

if not os.path.isdir(args.dataset):
	print '{} does not exist or is not a directory'.format(args.dataset)
	sys.exit()

n_input = 224
test_reader = ImageReader(os.path.abspath(args.dataset), "test.txt", 1, [n_input, n_input], False, False)

# tf Graph input
x = tf.placeholder(tf.float32, [None, n_input, n_input, 3], name="InputData")

# Define the network
poseNet = Posenet()
output = poseNet.create_testable(x)

saver = tf.train.Saver()
init = tf.initialize_all_variables()

if args.save_output:
	f = open(args.save_output, 'w')
	f.write('Prediction for {}\n'.format('test.txt'))
	f.write('ImageFile, Predicted Camera Position [X Y Z W P Q R]\n\n')

with tf.Session() as sess:
	sess.run(init)

	# Load the model
	saver.restore(sess, os.path.abspath(args.model))

	for i in range(test_reader.total_images()):
		images_feed, labels_feed = test_reader.next_batch()
		predicted = sess.run([output], feed_dict={x: images_feed})
		predicted = [round(v, 6) for v in predicted[0]['x'][0].tolist() + predicted[0]['q'][0].tolist()]
		gt = labels_feed[0]
		print "----{}----".format(i)
		print "Predicted: ", predicted
		print "Correct:   ", gt
		if args.save_output:
			x1, x2, x3 = predicted[0:3]
			q1, q2, q3, q4 = predicted[3:7]
			f.write('{0} {1:.6f} {2:.6f} {3:.6f} {4:.6f} {5:.6f} {6:.6f} {7:.6f}\n'.format(test_reader.images[i], x1, x2, x3, q1, q2, q3, q4))

if args.save_output:
	f.close()