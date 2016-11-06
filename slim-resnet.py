import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim.nets import resnet_v1
from image_queue import ImageReader
import matplotlib.pyplot as plt
import numpy as np
import argparse
import sys
import os

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

batch_size = 20
n_input = 224
learning_rate = 0.001
n_iters = 5000
n_disp = 5


train_log_dir = '/tmp/tensorflow_logs/resnet/run' + str(args.run)
if not tf.gfile.Exists(train_log_dir):
  tf.gfile.MakeDirs(train_log_dir)

# Prepare input queues
train_reader = ImageReader(os.path.abspath(args.imgdir), "dataset_train.txt", batch_size, [n_input, n_input])
test_reader = ImageReader(os.path.abspath(args.imgdir), "dataset_test.txt", 1, [n_input, n_input])

# tf Graph input
x = tf.placeholder(tf.float32, [None, n_input, n_input, 3], name="InputData")
y = tf.placeholder(tf.float32, [None, 7], name="LabelData")


# Define the model:
with slim.arg_scope(resnet_v1.resnet_arg_scope(is_training=True)):
	pred, _ = resnet_v1.resnet_v1_101(x, num_classes=None, global_pool=False, output_stride=None)
	shape = pred.get_shape().as_list()
	pred = tf.reshape(pred, [-1, shape[1]*shape[2]*shape[3]])
	pred = slim.fully_connected(pred, 7, activation_fn=None, weights_regularizer=slim.l2_regularizer(0.0005))

def split_label(output):
	return tf.slice(output, [0, 0], [-1, 3]), tf.nn.l2_normalize(tf.slice(output, [0, 3], [-1, 4]), 1)

pred_x, pred_q = split_label(pred)
real_x, real_q = split_label(y)

# Calculate cost
dx = tf.sqrt(tf.reduce_sum(tf.square(tf.sub(pred_x, real_x)), 1))
dq = tf.acos((2*tf.square(tf.reduce_sum(tf.mul(pred_q, real_q), 1)) - 1.0))
error_x = tf.reduce_mean(dx)
error_q = tf.reduce_mean(dq)


# Define loss and optimizer
cost = tf.reduce_mean(dx) + 2*tf.reduce_mean(dq)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)


# Initializing the variables
init = tf.initialize_all_variables()

# For saving the model
saver = tf.train.Saver()

# Define summary ops
tf.scalar_summary("loss", cost)
tf.scalar_summary("dx", error_x)
tf.scalar_summary("dq", error_q)
merged_summary_op = tf.merge_all_summaries()


#fig, ax = plt.subplots(4, 5)

# Launch the graph
with tf.Session() as sess:
	sess.run(init)

	if args.restore:
		print "Restoring the model..."
		restore_path = get_model_path(args.modeldir, args.run)
		saver.restore(sess, restore_path)
		print("Model restored from {}".format(restore_path))

	# Initialize the queue threads
	coord = tf.train.Coordinator()
	threads = tf.train.start_queue_runners(coord=coord)

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
				ex, eq, loss, summary = sess.run([error_x, error_q, cost, merged_summary_op], feed_dict={x: images_feed, y: labels_feed})
				summary_writer.add_summary(summary, i)
				print("Iteration " + str(i) + ", Loss= " + "{:.6f}".format(loss) \
					+ ", dx= " + "{:.6f}".format(ex) \
					+ ", dq= " + "{:.6f}".format(eq))
		print("Optimization Finished!")
		
		# Save the model
		if args.save:
			print "Saving the model..."
			save_path = saver.save(sess, get_model_path(args.modeldir, args.run))
			print("Model saved in file: %s" % save_path)

	if args.test:
		# Calculate accuracy for test images
		sum_ex = 0
		sum_eq = 0
		#fig, ax = plt.subplots(1, 1)
		for i in range(test_reader.total_batches()):
			print "----{}----".format(i)
			images_feed, labels_feed = test_reader.next_batch()
			p, ex, eq, loss = sess.run([pred, error_x, error_q, cost], feed_dict={x: images_feed, y: labels_feed})
			
			print "Predicted: ", p
			print "Correct:   ", labels_feed
			print "ex: {}".format(ex)
			print "eq: {}".format(eq)
			#plt.imshow(images_feed[0,:,:,:].astype(np.uint8))
			#plt.show()
			#for j in range(1):
			#	ax = plt.subplot(1, 1, j+1)
			#	a = images_feed[j,:,:,:]
			#	ax.imshow(a.astype(np.uint8))
			#plt.show()
			#sum_ex += ex
			#sum_eq += eq
		print "----------------------"
		print "Mean error x: {}".format(sum_ex/test_reader.total_batches())
		print "Mean error q: {}".format(sum_eq/test_reader.total_batches())

	coord.request_stop()
	coord.join(threads)

	