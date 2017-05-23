import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.ticker as ticker
import matplotlib.image as image
from matplotlib.patches import Wedge
import numpy as np

from posenet.core.image_reader import read_label_file
from posenet.utils import rotate_by_quaternion

def draw_plane(a):
	ax.add_patch(patches.Rectangle((-a, -a), a, a, facecolor='blue', linewidth=0))
	ax.add_patch(patches.Rectangle((0, -a), a, a, facecolor=(0.9,0.9,0.9), linewidth=0))
	ax.add_patch(patches.Rectangle((0, 0), a, a, facecolor='green', linewidth=0))
	ax.add_patch(patches.Rectangle((-a, 0), a, a, facecolor='red', linewidth=0))

def draw_circle(radius, pos=(0,0), color='k', lw=1, theta=None):
	if theta is None:
		theta = np.linspace(0, 2*np.pi, 64)
	plt.plot(pos[0] + radius*np.cos(theta), pos[1] + radius*np.sin(theta), color, lw=lw)
	t = np.linspace(0, 2*np.pi, 21)
	plt.plot(pos[0] + radius*np.cos(t), pos[1] + radius*np.sin(t), color, marker='.', lw=0, ms=10)

def_file = 'results_new/david_6a/test2_localised.txt'
david = True
side = True
extrapolation = False
radius = 3.5#*np.sin(1.2)
lim = 4.5

_, labels = read_label_file(def_file, convert_to_axis=False)
positions = np.array([l[0:3] for l in labels])
orientations = np.array([l[3:] for l in labels])

if side and david:
	positions_2d = [(np.sqrt(p[0]*p[0] + p[1]*p[1]), p[2]) for p in positions]
	orientations_2d = [(-1,0) for o in orientations]
elif side:
	positions_2d = [((p[0]+p[1])/np.sqrt(2), p[2]) for p in positions]
	orientations_2d = [((o[0]+o[1])/np.sqrt(2), o[2]) for o in orientations]
else:
	positions_2d = [(p[0], p[1]) for p in positions]
	orientations_2d = [(o[0], o[1]) for o in orientations]

plt.figure(figsize=(4,4), dpi=80, facecolor='w')
ax = plt.axes()

if side:
	if david:
		im = image.imread('results_new/misc/david.png')
		ax = plt.axes()
		ax.imshow(im, aspect='auto', extent=(-1.6,1.6,-0.6,6.5), zorder=10)
		plt.plot([3.5,3.5], [0.5,5.997787], 'k', lw=1)
		plt.plot([3.5]*6, np.linspace(0.5,5.997787,6), 'k', marker='.', lw=0, ms=10)
	else:
		draw_circle(3.5, theta=-np.linspace(0.1, 3.0416, 100)+np.pi/2)
		plt.plot([-0.8, -0.4], [0,0], color=(0.9,0.9,0.9))
		plt.plot([-0.4, 0.4], [0,0], color='green')
		plt.plot([0.4, 0.8], [0,0], color='red')
else:
	if extrapolation:
		draw_circle(radius, theta=np.linspace(np.pi/2,np.pi*3/2))
		ax.add_patch(patches.Wedge((0, 0), radius, 90, 150,
			facecolor=(0.7,0.7,0.7), alpha=0.5, edgecolor='none'))
		ax.add_patch(patches.Wedge((0, 0), radius, 210, 270,
			facecolor=(0.7,0.7,0.7), alpha=0.5, edgecolor='none'))
	else:
		draw_circle(radius)
	if david:
		im = image.imread('results_new/misc/david_top.png')
		ax = plt.axes()
		ax.imshow(im, aspect='auto', extent=(-2.2,1.9,-2.2,1.9), zorder=10)
	else:
		draw_plane(0.8)
		#draw_circle(1, pos=(radius-1,0.7), color='red', lw=1.5)
		#draw_circle(1, pos=(0.2,radius), color='red', lw=1.5)

if extrapolation:
	n = len(positions_2d)
	positions_2d = positions_2d[n/4:-n/4]
	orientations_2d = orientations_2d[n/4:-n/4]

for pos, orient in zip(positions_2d, orientations_2d):
	ax.arrow(pos[0], pos[1], orient[0], orient[1], 
		head_width=0.06, head_length=0.15, fc='k', ec='k', zorder=10)

#plt.xlim(-lim,lim)
#plt.ylim(-lim,lim)
plt.xlim(-1,3)
plt.ylim(-0.5,7)
ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
ax.yaxis.set_major_locator(ticker.MultipleLocator(2))
ax.set_aspect('equal', 'datalim')
plt.title('Test 2, overtrained')
plt.savefig('results_new/david_6a/test2_60k.pdf')
