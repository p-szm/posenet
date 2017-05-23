import itertools
import math

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker

from posenet.core.image_reader import read_label_file

def_file = 'datasets/plane/test1.txt'
test_file = 'results_new/plane_axis/test1_localised.txt'
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
elif flip == 'smooth':
	for i, x in enumerate(orientations_gt[:-1]):
		if sum(abs(orientations_gt[i+1] - x)) > sum(abs(orientations_gt[i+1] + x)):
			orientations_gt[i+1] = -orientations_gt[i+1]
elif flip is not None:
	orientations_gt[flip[0]:flip[1]] = -orientations_gt[flip[0]:flip[1]]

x = np.linspace(0, 2*np.pi, orientations_gt.shape[0])

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
	plt.figure(figsize=(4, 3.5), dpi=80, facecolor='w')
	plt.plot(x/np.pi, orientations_gt, color='grey')
	plt.plot(x/np.pi, orientations_test, lw=1)
	plt.axhline(0, color='grey', ls='dashed')
	plt.title('Orientation')
	plt.xlabel('Angular position')
	plt.ylabel('Vector components')
	ax = plt.axes()
	ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('$%g\pi$'))
	ax.xaxis.set_major_locator(ticker.MultipleLocator(base=0.5))
	plt.subplots_adjust(bottom=0.15, left=0.18)
	plt.savefig('results_new/plane_all/a.pdf')

	plt.figure(figsize=(4, 3.5), dpi=80, facecolor='w')
	plt.plot(x/np.pi, positions_gt, color='grey', lw=1)
	plt.plot(x/np.pi, positions_test)
	plt.title('Position')
	plt.xlabel('Angular position')
	plt.ylabel('Vector components')
	ax = plt.axes()
	ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('$%g\pi$'))
	ax.xaxis.set_major_locator(ticker.MultipleLocator(base=0.5))
	plt.subplots_adjust(bottom=0.15, left=0.18)
	plt.savefig('results_new/plane_all/b.pdf')
