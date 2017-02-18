import argparse

import matplotlib.colors as colors
import matplotlib.cm as cmx
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

from posenet.core.image_reader import ImageReader, read_label_file
from posenet.core.localiser import Localiser
from posenet.utils import progress_bar
from posenet.utils import rotate_by_quaternion


parser = argparse.ArgumentParser(description='''
    Visualise camera poses given a definition file''')
parser.add_argument('-d', '--dataset', action='store', required=True, 
    help='''Path to a text file listing images and camera poses''')
parser.add_argument('-m', '--model', action='store', required=False, 
    help=''''Model used to predict camera poses. If not specified, real
    camera poses will be plotted''')
parser.add_argument('--sphere_pos', action='store', type=float, nargs=3, default=[0,0,0])
parser.add_argument('--r_sphere', action='store', type=float, required=False, default=1)
parser.add_argument('--arrow_len', action='store', type=float, required=False, default=1)
parser.add_argument('--connect', action='store_true',
    help='''Connect consecutive camera positions with lines''')
parser.add_argument('--rings', action='store', nargs='*', required=False)
parser.add_argument('--plot_gt', action='store_true')
parser.add_argument('--plot_diff', action='store_true')
parser.add_argument('-u', '--uncertainty', action='store_true')
args = parser.parse_args()

if not args.model or args.plot_gt or args.plot_diff:
    _, labels = read_label_file(args.dataset)
    positions_gt = np.array([l[0:3] for l in labels])
    orientations_gt = np.array([l[3:7] for l in labels])

if args.model:
    input_size = 256
    test_reader = ImageReader(args.dataset, batch_size=1, 
                        image_size=[input_size, input_size], randomise=False)
    n_images = test_reader.total_images()

    positions = np.empty([0,3])
    orientations = np.empty([0,4])
    if args.uncertainty:
        std_x = []
        std_q = []

    with Localiser(args.model, uncertainty=args.uncertainty) as localiser:
        for i in range(n_images):
            images_feed, labels_feed = test_reader.next_batch()

            # Make prediction
            predicted = localiser.localise(images_feed)
            positions = np.concatenate((positions, np.asarray([predicted['x']])))
            orientations = np.concatenate((orientations, np.asarray([predicted['q']])))
            if args.uncertainty:
                std_x.append(predicted['std_x'])
                std_q.append(predicted['std_q'])

            progress_bar(1.0*(i+1)/n_images, 30, text='Localising')
        print('')
else:
    positions = positions_gt
    orientations = orientations_gt


def draw_segment(start, end, color='black', lw=1):
    plt.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], 
        linestyle='-', color=color, lw=lw)

def draw_sphere(pos, r, color='black', lw=1):
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = pos[0] + r * np.outer(np.cos(u), np.sin(v))
    y = pos[1] + r * np.outer(np.sin(u), np.sin(v))
    z = pos[2] + r * np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_wireframe(x, y, z, rstride=4, cstride=4, color=color, lw=lw)

def draw_ring(pos, r, color='black', lw=1):
    r = float(r)
    phi = np.linspace(-np.pi, np.pi, 100)
    x = pos[0] + r*np.cos(phi)
    y = pos[1] + r*np.sin(phi)
    z = pos[2] + np.zeros(100)
    ax.plot(x, y, z, color=color, lw=lw)

# Plot setup
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Draw coordinates
p0 = args.sphere_pos
l = 2 * args.r_sphere
draw_segment(p0, [p0[0]+l, p0[1], p0[2]], color='red', lw=1.5)
draw_segment(p0, [p0[0], p0[1]+l, p0[2]], color='green', lw=1.5)
draw_segment(p0, [p0[0], p0[1], p0[2]+l], color='blue', lw=1.5)

# Draw sphere
draw_sphere(args.sphere_pos, args.r_sphere, color='black', lw=0.5)

# Draw rings at z=0
if args.rings:
    for R in args.rings:
        draw_ring(args.sphere_pos, R, color='black', lw=0.5)

# Define color scheme for arrows
if args.uncertainty:
    cmap = plt.cm.jet
    cNorm  = colors.Normalize(vmin=0, vmax=max(a+b for a, b in zip(std_x, std_q)))
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cmap)

# Draw arrows
vec = np.repeat(np.array([[0,0,-1.0]]), positions.shape[0], axis=0)
for i in range(vec.shape[0]):
    vec[i,:] = rotate_by_quaternion(vec[i,:], orientations[i,:])
arrows = np.concatenate((positions, vec), axis=1).T
X,Y,Z,U,V,W = arrows
for i in range(positions.shape[0]):
    if args.uncertainty:
        colorVal = scalarMap.to_rgba(std_x[i]+std_q[i])
    else:
        colorVal = 'red'
    ax.quiver(X[i],Y[i],Z[i],U[i],V[i],W[i], pivot='tail', color=colorVal, length=args.arrow_len, lw=1)

# Connect consecutive predicted positions
if args.connect:
    ax.plot(positions[:,0], positions[:,1], positions[:,2], color='red')

# Draw ground truth positions (connected)
if args.plot_gt:
    ax.plot(positions_gt[:,0], positions_gt[:,1], positions_gt[:,2], color='black')

# Draw lines from gt positions to predicted positions
if args.plot_diff:
    for i in range(positions.shape[0]): # Yeah I know it's slow
        draw_segment(positions_gt[i][:], positions[i][:], color=(0.5, 0.5, 0.5))

# Set limits
R = np.max(np.linalg.norm(positions, axis=1))
ax.set_xlim(-R, R)
ax.set_ylim(-R, R)
ax.set_zlim(-R, R)

# Camera position
ax.view_init(elev=30, azim=-60)
ax.dist=6

# No axes
ax.set_axis_off()
plt.show()