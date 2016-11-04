import tensorflow as tf
import os
#import matplotlib.pyplot as plt
#import matplotlib.image as mpimg
import numpy as np


img_dir = "/home/pszmucer/workspace/resnet/ShopFacade/"

def read_label_file(dir, def_file):
	paths = []
	labels = []
	with open(os.path.join(img_dir, def_file)) as f:
		lines = map(lambda line: line.rstrip("\n").split(" "), f.readlines()[3:])
		paths = map(lambda line: os.path.join(img_dir, line[0]), lines)
		labels = map(lambda line: map(lambda x: float(x), line[1:]), lines)
	return paths, labels

def load_from_queue(queue):
	label = queue[1]
	img = tf.image.decode_png(tf.read_file(queue[0]), channels=3)
	return img, label

def create_input_pipeline(filepaths, labels, batch_size):
	all_images = tf.convert_to_tensor(filepaths, dtype=tf.string)
	all_labels = tf.convert_to_tensor(labels, dtype=tf.float32)

	input_queue = tf.train.slice_input_producer([filepaths, labels], shuffle=False)

	image = tf.image.decode_png(tf.read_file(input_queue[0]), channels=3)
	label = input_queue[1]

	image.set_shape([1080, 1920, 3])
	#image = tf.random_crop(image, [900, 900, 3])
	image = tf.image.resize_images(image, [224, 224])

	return tf.train.batch([image, label], batch_size=batch_size)


'''create_input_pipeline(img_dir, "dataset_train.txt", batch_size)

print "input pipeline ready"

with tf.Session() as sess:
  # initialize the variables
  sess.run(tf.initialize_all_variables())
  
  # initialize the queue threads to start to shovel data
  coord = tf.train.Coordinator()
  threads = tf.train.start_queue_runners(coord=coord)

  print "from the train set:"
  for i in range(20):
    b = train_image_batch.eval()
    a = b[0,:,:,:]
    plt.imshow(a.astype(np.uint8))
    plt.show()

  # stop our queue threads and properly close the session
  coord.request_stop()
  coord.join(threads)
  sess.close()'''