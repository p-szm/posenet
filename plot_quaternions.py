import itertools
import math

import matplotlib.pyplot as plt
import numpy as np

from posenet.core.image_reader import read_label_file

def_file = 'datasets/plane/test1.txt'
test_file = 'results/localised/plane_test1.txt'
#flip = [41,None]
#flip = [87, 135]
flip = 'auto'
subplots = False

def plot_orientation(ax, i):
	ax.plot(orientations_gt[:,i], color='grey')
	ax.plot(orientations_test[:,i])
	ax.yaxis.set_ticks([-1,0,1])
	ax.axhline(0, color='grey', ls='dashed')
	#ax.set_title('q{}'.format(i))

def plot_position(ax, i):
	ax.plot(positions_gt[:,i], color='grey')
	ax.plot(positions_test[:,i])
	#ax.yaxis.set_ticks(np.arange(min_pos, max_pos, 1))
	ax.set_ylim(min(0, math.floor(min(min(positions_gt[:,i]), min(positions_test[:,i])))-1))


_, labels_gt = read_label_file(def_file, convert_to_axis=True)
_, labels_test = read_label_file(test_file, convert_to_axis=False)

positions_gt = np.array(map(lambda x: x[0:3], labels_gt))
positions_test = np.array(map(lambda x: x[0:3], labels_test))

orientations_gt = np.array(map(lambda x: x[3:], labels_gt))
orientations_test = np.array(map(lambda x: x[3:], labels_test))

if orientations_gt.shape[1] == 3:
	pass
elif flip == 'auto':
	for i, x in enumerate(orientations_gt):
		if sum(abs(-x - orientations_test[i])) < sum(abs(x - orientations_test[i])):
			orientations_gt[i] = -x
elif flip is not None:
	orientations_gt[flip[0]:flip[1]] = -orientations_gt[flip[0]:flip[1]]

if subplots:
	k = orientations_gt.shape[1]
	f, ax = plt.subplots(k, 1, sharex=True)
	for i in range(k):
		plot_orientation(ax[i], i)
	plt.title('Orientation')
	#f.subplots_adjust(hspace=0)

	f, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)
	plot_position(ax1, 0)
	plot_position(ax2, 1)
	plot_position(ax3, 2)
	plt.title('Position')
	plt.show()
else:
	plt.plot(orientations_gt, color='grey')
	plt.plot(orientations_test)
	plt.axhline(0, color='grey', ls='dashed')
	plt.title('Orientation')
	plt.figure()
	plt.plot(positions_gt, color='grey')
	plt.plot(positions_test)
	plt.title('Position')
	plt.show()
