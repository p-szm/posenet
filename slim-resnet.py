import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim.nets import resnet_v1
from image_queue import ImageReader
import matplotlib.pyplot as plt
import numpy as np
import argparse
import sys
import os
from posenet import Posenet

parser = argparse.ArgumentParser()
parser.add_argument('--test', action='store_true')
parser.add_argument('--train', action='store_true')
parser.add_argument('--restore', action='store_true')
parser.add_argument('--save', action='store_true')
parser.add_argument('--run', action='store', type=int)
parser.add_argument('--imgdir', action='store')
parser.add_argument('--modeldir', action='store')

args = parser.parse_args()

if args.run == None:
	print "Run number not defined"
	sys.exit()

if args.imgdir == None:
	print "Image directory not defined"
	sys.exit()

if args.modeldir == None and (args.save or args.restore):
	print "Model directory not defined"
	sys.exit()

def get_model_path(model_dir, run):
	return os.path.join(os.path.abspath(model_dir), "model" + str(run) + ".ckpt")

batch_size = 32
test_batch_size = 1
n_input = 224
learning_rate = 0.001
n_iters = 100
n_disp = 5


train_log_dir = '/tmp/tensorflow_logs/resnet/run' + str(args.run)
if not tf.gfile.Exists(train_log_dir):
	tf.gfile.MakeDirs(train_log_dir)

# Prepare input queues
train_reader = ImageReader(os.path.abspath(args.imgdir), "dataset_train.txt", batch_size, [n_input, n_input])
test_reader = ImageReader(os.path.abspath(args.imgdir), "dataset_test.txt", test_batch_size, [n_input, n_input])

# tf Graph input
x = tf.placeholder(tf.float32, [None, n_input, n_input, 3], name="InputData")
y = tf.placeholder(tf.float32, [None, 7], name="LabelData")

# Define the network
poseNet = Posenet()
cost, output = poseNet.create_trainable(x, y)
test_output, _ = poseNet.create_testable(x, reuse=True)

# Define the optimiser
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Initializing the variables
init = tf.initialize_all_variables()

# For saving the model
saver = tf.train.Saver()

# Define summary ops
merged_summary_op = tf.merge_all_summaries()

# Launch the graph
with tf.Session() as sess:
	sess.run(init)

	if args.restore:
		print "Restoring the model..."
		restore_path = get_model_path(args.modeldir, args.run)
		saver.restore(sess, restore_path)
		print("Model restored from {}".format(restore_path))

	if args.train:
		# op to write logs to Tensorboard
		summary_writer = tf.train.SummaryWriter(train_log_dir, graph=tf.get_default_graph())

		for i in range(n_iters):
			images_feed, labels_feed = train_reader.next_batch()

			#for j in range(20):
			#	ax = plt.subplot(4, 5, j+1)
			#	a = images_feed[j,:,:,:]
			#	ax.imshow(a.astype(np.uint8))
			#plt.show()

			# Run optimization op (backprop)
			sess.run([optimizer], feed_dict={x: images_feed, y: labels_feed})

			if (i % n_disp == 0):
				loss, summary = sess.run([cost, merged_summary_op], feed_dict={x: images_feed, y: labels_feed})
				summary_writer.add_summary(summary, i)
				print("Iteration " + str(i) + ", Loss= " + "{:.6f}".format(loss))
		print("Optimization Finished!")
		
		# Save the model
		if args.save:
			print "Saving the model..."
			save_path = saver.save(sess, get_model_path(args.modeldir, args.run))
			print("Model saved in file: %s" % save_path)

	if args.test:
		fig, ax = plt.subplots(1, test_batch_size)
		for i in range(test_reader.total_batches()):
			print "----{}----".format(i)
			images_feed, labels_feed = test_reader.next_batch()

			#for j in range(images_feed.shape[0]):
			#	print 'Img mean: ', np.mean(images_feed[j,:,:,:])

			output = sess.run([test_output], feed_dict={x: images_feed, y: labels_feed})

			#for j in range(v.shape[0]):
			#	print np.mean(v[j,:,:,:])
			
			print "Predicted: ", output
			print "Correct:   ", labels_feed

			'''for j in range(test_batch_size):
				ax = plt.subplot(1, test_batch_size, j+1)
				a = images_feed[j,:,:,:]
				ax.imshow(a.astype(np.uint8))
			plt.show()'''

