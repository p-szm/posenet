import argparse

import numpy as np

from image_reader import ImageReader
from localiser import Localiser
from utils import l2_distance, quaternion_distance, rotate_by_quaternion


parser = argparse.ArgumentParser()
parser.add_argument('--model', action='store', required=True)
parser.add_argument('--dataset', action='store', required=True)
parser.add_argument('--plot_errors', action='store', nargs='?', const='')
parser.add_argument('--plot_3d', action='store', nargs='?', const='')
args = parser.parse_args()


input_size = 224
test_reader = ImageReader(args.dataset, batch_size=1, 
						image_size=[input_size, input_size], 
						random_crop=False, randomise=False)

pos_errors = []
orient_errors = []
positions = np.empty([0,3])
orientations = np.empty([0,4])

with Localiser(input_size, args.model) as localiser:
	print 'Localising...'
	for i in range(test_reader.total_images()):
		images_feed, labels_feed = test_reader.next_batch()
		gt = {'x': labels_feed[0][0:3], 'q': labels_feed[0][3:7]}

		# Make prediction
		predicted = localiser.localise(images_feed)
		x, y, z = predicted['x']
		q1, q2, q3, q4 = predicted['q']
		print predicted

		pos_error = l2_distance(gt['x'], predicted['x'])
		orient_error = quaternion_distance(gt['q'], predicted['q']) * 180 / np.pi
		positions = np.concatenate((positions, np.asarray([[x, y, z]])))
		orientations = np.concatenate((orientations, np.asarray([[q1, q2, q3, q3]])))

		pos_errors.append(pos_error)
		orient_errors.append(orient_error)

		print('-------------{}-------------'.format(i))
		print('Positional error:  {}'.format(pos_error))
		print('Orientation error: {}'.format(orient_error))
	print 'Done'


if args.plot_errors is not None:
	import matplotlib.pyplot as plt

	x = np.linspace(-90, 90, test_reader.total_images())

	fig, axes = plt.subplots(nrows=2, ncols=1)
	fig.patch.set_facecolor('white')
	ax1 = axes[0]
	ax2 = axes[1]

	ax1.plot(x, pos_errors, color='black')
	ax1.set_xlim([-90, 90])
	ax1.axvspan(-90, -45, alpha=0.2, color='red')
	ax1.axvspan(45, 90, alpha=0.2, color='red')
	ax1.set_ylabel("Positional error")

	ax2.plot(x, orient_errors, color='black')
	ax2.set_xlim([-90, 90])
	ax2.axvspan(-90, -45, alpha=0.2, color='red')
	ax2.axvspan(45, 90, alpha=0.2, color='red')
	ax2.set_xlabel('Phi')
	ax2.set_ylabel("Orientation error (degrees)")

	if args.plot_errors:
		plt.savefig(args.plot_errors)
	else:
		plt.show()

if args.plot_3d is not None:
	from mpl_toolkits.mplot3d import Axes3D
	import matplotlib.pyplot as plt

	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')

	# Sphere
	r_sphere = 3
	u = np.linspace(0, 2 * np.pi, 100)
	v = np.linspace(0, np.pi, 100)
	x = r_sphere * np.outer(np.cos(u), np.sin(v))
	y = r_sphere * np.outer(np.sin(u), np.sin(v))
	z = r_sphere * np.outer(np.ones(np.size(u)), np.cos(v))
	ax.plot_wireframe(x, y, z, rstride=4, cstride=4, color='b')

	# Path
	ax.plot(positions[:,0], positions[:,1], positions[:,2], 
			label='parametric curve', color='red')

	# Arrows
	vec = np.repeat(np.array([[0,0,-1.0]]), positions.shape[0], axis=0)
	for i in xrange(vec.shape[0]):
		vec[i,:] = rotate_by_quaternion(vec[i,:], orientations[i,:])
	arrows = np.concatenate((positions, vec), axis=1).T
	X,Y,Z,U,V,W = arrows
	ax.quiver(X,Y,Z,U,V,W, pivot='tail', color='green', length=1.5)

	# Plot limits
	R = 1.1*np.max(np.linalg.norm(positions, axis=1))
	ax.set_xlim(-R, R)
	ax.set_ylim(-R, R)
	ax.set_zlim(-R, R)

	if args.plot_3d:
		plt.savefig(args.plot_3d)
	else:
		plt.show()