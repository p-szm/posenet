import tensorflow as tf
import tensorflow.contrib.slim as slim
import matplotlib.pyplot as plt
import numpy as np
import argparse
import sys
import os
from posenet import Posenet
from image_reader import ImageReader

parser = argparse.ArgumentParser()
parser.add_argument('--restore', action='store')
parser.add_argument('--save', action='store')
parser.add_argument('--run', action='store', type=int)
parser.add_argument('--imgdir', action='store')

args = parser.parse_args()

if args.run == None:
	print "Run number not defined"
	sys.exit()

if args.imgdir == None:
	print "Image directory not defined"
	sys.exit()

def get_save_path(path, run):
	return os.path.join(os.path.abspath(path), "model" + str(run) + ".ckpt")

batch_size = 32
validation_batch_size = 32
n_input = 224
learning_rate = 0.001
n_iters = 5000
n_disp = 5
n_disp_validation = 30
beta = 20


log_dir = '/home/pszmucer/workspace/resnet/runs/run' + str(args.run)
if not tf.gfile.Exists(log_dir):
	tf.gfile.MakeDirs(log_dir)

# Prepare input queues
train_reader = ImageReader(os.path.abspath(args.imgdir), "training.txt", batch_size, [n_input, n_input], False, True)
validation_reader = ImageReader(os.path.abspath(args.imgdir), "validation.txt", validation_batch_size, [n_input, n_input], False, True)

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

	for i in range(n_iters):
		train_images_feed, train_labels_feed = train_reader.next_batch()

		#for j in range(20):
		#	ax = plt.subplot(4, 5, j+1)
		#	a = images_feed[j,:,:,:]
		#	ax.imshow(a.astype(np.uint8))
		#plt.show()

		# Run optimization op (backprop)
		sess.run([optimizer], feed_dict={x: train_images_feed, y: train_labels_feed})

		if (i % n_disp == 0):
			print "----- Iter {} -----".format(i)
			results = sess.run(
				[train_loss]+train_summaries, feed_dict={x: train_images_feed, y: train_labels_feed})
			for res in results[1:]:
				summary_writer.add_summary(res, i)
			print("(TRAIN) Loss= " + "{:.6f}".format(results[0]))

		if (i % n_disp_validation == 0):
			val_images_feed, val_labels_feed = validation_reader.next_batch()
			results = sess.run(
				[validation_loss]+validation_summaries, feed_dict={x: val_images_feed, y: val_labels_feed})
			for res in results[1:]:
				summary_writer.add_summary(res, i)
			print("(VALIDATION)  Loss= " + "{:.6f}".format(results[0]))

	print "-----------------------------"
	print("Optimization Finished!")
	
	# Save the model
	if args.save:
		print "Saving the model..."
		save_path = saver.save(sess, get_save_path(args.save, args.run))
		print("Model saved in file: %s" % save_path)

