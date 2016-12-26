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
args = parser.parse_args()

if not os.path.isfile(args.model) or os.path.isdir(args.model):
	print '{} does not exist or is a directory'.format(args.model)
	sys.exit()

if not os.path.isdir(args.dataset):
	print '{} does not exist or is not a directory'.format(args.dataset)
	sys.exit()

n_input = 224
test_reader = ImageReader(os.path.abspath(args.dataset), "test.txt", 1, [n_input, n_input], False)

# tf Graph input
x = tf.placeholder(tf.float32, [None, n_input, n_input, 3], name="InputData")

# Define the network
poseNet = Posenet()
output = poseNet.create_testable(x)

saver = tf.train.Saver()
init = tf.initialize_all_variables()

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
