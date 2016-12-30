import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

from image_reader import ImageReader
from posenet import Posenet


parser = argparse.ArgumentParser(description='''
	Train the PoseNet network''')
parser.add_argument('--dataset', action='store', required=True,
	help='''Path to the definition file used for training''')
parser.add_argument('--validate', action='store', required=True,
	help='''Path to the definition file used for validation''')
parser.add_argument('--logdir', action='store', default='runs',
	help='''Path the the directory to which logs will be saved''')
parser.add_argument('--run', action='store', type=int, required=True,
	help='''Run number. Will be used to name the saved model and log file''')
parser.add_argument('--save_dir', action='store', default='models',
	help='''Directory in which the model will be saved''')
parser.add_argument('--restore', action='store',
	help='''Path to a model which will be restored''')
parser.add_argument('--batch_size', action='store', type=int, default=32,
	help='''Batch size for training and validation''')
parser.add_argument('--n_iters', action='store', type=int, default=5000,
	help='''Number of iterations for which training will be performed''')
parser.add_argument('--n_disp', action='store', type=int, default=5,
	help='''Calculate training accuracy every nth iteration''')
parser.add_argument('--n_disp_validation', action='store', type=int, default=20,
	help='''Calculate validation accuracy every nth iteration''')
args = parser.parse_args()


n_input = 224
learning_rate = 0.001
beta = 20

log_dir = os.path.join(args.logdir, 'run{}'.format(args.run))
if not tf.gfile.Exists(log_dir):
	tf.gfile.MakeDirs(log_dir)
if not tf.gfile.Exists(args.save_dir):
	tf.gfile.MakeDirs(args.save_dir)

# Prepare input queues
train_reader = ImageReader(args.dataset, args.batch_size, [n_input, n_input], False, True)
validation_reader = ImageReader(args.validate, args.batch_size, [n_input, n_input], False, True)

# tf Graph input
x = tf.placeholder(tf.float32, [None, n_input, n_input, 3], name="InputData")
y = tf.placeholder(tf.float32, [None, 7], name="LabelData")

# Define the network
poseNet = Posenet()
train_output, train_loss, train_summaries = poseNet.create_trainable(x, y, beta=beta)
validation_output, validation_loss, validation_summaries = poseNet.create_validation(x, y, beta=beta, reuse=True)

# Define the optimiser
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(train_loss)

# Initializing the variables
init = tf.initialize_all_variables()

# For saving the model
saver = tf.train.Saver()

# Launch the graph
with tf.Session() as sess:
	sess.run(init)

	if args.restore:
		print "Restoring the model..."
		saver.restore(sess, os.path.abspath(args.restore))
		print("Model restored from {}".format(args.restore))

	# op to write logs to Tensorboard
	summary_writer = tf.train.SummaryWriter(log_dir, graph=tf.get_default_graph())

	for i in range(args.n_iters):
		train_images_feed, train_labels_feed = train_reader.next_batch()

		#for j in range(20):
		#	ax = plt.subplot(4, 5, j+1)
		#	a = images_feed[j,:,:,:]
		#	ax.imshow(a.astype(np.uint8))
		#plt.show()

		# Run optimization op (backprop)
		sess.run([optimizer], feed_dict={x: train_images_feed, y: train_labels_feed})

		if (i % args.n_disp == 0):
			print "----- Iter {} -----".format(i)
			results = sess.run(
				[train_loss]+train_summaries, feed_dict={x: train_images_feed, y: train_labels_feed})
			for res in results[1:]:
				summary_writer.add_summary(res, i)
			print("(TRAIN) Loss= " + "{:.6f}".format(results[0]))

		if (i % args.n_disp_validation == 0):
			val_images_feed, val_labels_feed = validation_reader.next_batch()
			results = sess.run(
				[validation_loss]+validation_summaries, feed_dict={x: val_images_feed, y: val_labels_feed})
			for res in results[1:]:
				summary_writer.add_summary(res, i)
			print("(VALIDATION)  Loss= " + "{:.6f}".format(results[0]))

	print "-----------------------------"
	print("Optimization Finished!")
	
	# Save the model
	print "Saving the model..."
	save_path = os.path.join(args.save_dir, 'model{}.ckpt'.format(args.run))
	saver.save(sess, save_path)
	print("Model saved in file: %s" % save_path)

