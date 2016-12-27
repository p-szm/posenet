import tensorflow as tf
from posenet import Posenet
import numpy

class Localiser:
	def __init__(self, input_size, model_path):
		# Define the network
		self.x = tf.placeholder(tf.float32, [None, input_size, input_size, 3], name="InputData")
		self.network = Posenet()
		self.output = self.network.create_testable(self.x)
		self.model_path = model_path

		# Initialise other stuff
		self.saver = tf.train.Saver()
		self.init = tf.initialize_all_variables()
		self.session = None

	def __enter__(self):
		self.session = tf.Session()
		self.session.run(self.init)
		self.saver.restore(self.session, self.model_path) # Load the model
		return self

	def __exit__(self, exc_type, exc_value, traceback):
		self.session.close()

	def localise(self, img):
		"""Accepts a numpy image [size, size, n_channels] or [1, size, size, n_channels]"""
		if len(img.shape) == 3:
			img = np.expand_dims(img, axis=0)
		predicted = self.session.run([self.output], feed_dict={self.x: img})
		return {'x': predicted[0]['x'][0].tolist(), 'q': predicted[0]['q'][0].tolist()}
