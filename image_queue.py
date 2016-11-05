import matplotlib.image as img
import numpy as np
import scipy.misc
from random import randint, shuffle
import math
import os

def read_label_file(def_file):
	paths = []
	labels = []
	with open(def_file) as f:
		lines = map(lambda line: line.rstrip("\n").split(" "), f.readlines()[3:])
		paths = map(lambda line: line[0], lines)
		labels = map(lambda line: map(lambda x: float(x), line[1:]), lines)
	return paths, labels

class ImageReader:
	def __init__(self, image_dir, def_file, batch_size, image_size):
		self.image_dir = image_dir
		self.batch_size = batch_size
		self.image_size = image_size
		self.images, self.labels = read_label_file(self._full_path(def_file))
		self.idx = 0

	def _full_path(self, f):
		return os.path.join(self.image_dir, f)

	def _reset(self, randomise):
		self.idx = 0
		if randomise:
			index_shuf = range(len(self.images))
			shuffle(index_shuf)
			self.images = [self.images[i] for i in index_shuf]
			self.labels = [self.labels[i] for i in index_shuf]

	def _read_image(self, image_path):
		image = img.imread(self._full_path(image_path))
		# Random square crop
		side = min(image.shape[0], image.shape[1])
		side = max(int(side*0.9), min(*self.image_size))
		tr_x = randint(0, image.shape[1]-side)
		tr_y = randint(0, image.shape[0]-side)
		image = image[tr_y:side+tr_y,tr_x:side+tr_x]
		image = scipy.misc.imresize(image, [self.image_size[0], self.image_size[1], 3])
		return image

	def next_batch(self):
		images = map(lambda image: self._read_image(image), 
					self.images[self.idx:self.idx+self.batch_size])
		images = np.asarray(images)
		labels = self.labels[self.idx:self.idx+self.batch_size]
		self.idx += self.batch_size
		if self.idx >= len(self.images):
			self._reset(True)
		return images, labels

	def total_images(self):
		return len(self.images)

	def total_batches(self):
		return int(math.ceil(len(self.images)/self.batch_size))