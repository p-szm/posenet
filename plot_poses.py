import argparse

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

from posenet.core.image_reader import ImageReader, read_label_file
from posenet.core.localiser import Localiser
from posenet.utils import progress_bar
from posenet.utils import rotate_by_quaternion


parser = argparse.ArgumentParser(description='''
    Visualise camera poses given a definition file''')
parser.add_argument('--dataset', action='store', required=True, 
    help='''Path to a text file listing images and camera poses''')
parser.add_argument('--model', action='store', required=False, 
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
args = parser.parse_args()

if not args.model or args.plot_gt or args.plot_diff:
    _, labels = read_label_file(args.dataset)
    positions_gt = np.array([l[0:3] for l in labels])
    orientations_gt = np.array([l[3:7] for l in labels])

if args.model:
    input_size = 224
    test_reader = ImageReader(args.dataset, batch_size=1, 
                        image_size=[input_size, input_size], 
                        random_crop=False, randomise=False)
    n_images = test_reader.total_images()

    positions = np.empty([0,3])
    orientations = np.empty([0,4])

    with Localiser(input_size, args.model) as localiser:
        for i in range(n_images):
            images_feed, labels_feed = test_reader.next_batch()

            # Make prediction
            predicted = localiser.localise(images_feed)
            positions = np.concatenate((positions, np.asarray([predicted['x']])))
            orientations = np.concatenate((orientations, np.asarray([predicted['q']])))

            progress_bar(1.0*(i+1)/n_images, 30, text='Localising')
        print('')
else:
    positions = positions_gt
    orientations = orientations_gt


# Plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Sphere
u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, np.pi, 100)
x = args.sphere_pos[0] + args.r_sphere * np.outer(np.cos(u), np.sin(v))
y = args.sphere_pos[1] + args.r_sphere * np.outer(np.sin(u), np.sin(v))
z = args.sphere_pos[2] + args.r_sphere * np.outer(np.ones(np.size(u)), np.cos(v))
ax.plot_wireframe(x, y, z, rstride=4, cstride=4, color='black', lw=0.5)

if args.rings:
    phi = np.linspace(-np.pi, np.pi, 100)
    z = args.sphere_pos[2] + np.zeros(100)
    for R in args.rings:
        R = float(R)
        x = args.sphere_pos[0] + R*np.cos(phi)
        y = args.sphere_pos[1] + R*np.sin(phi)
        ax.plot(x, y, z, color=(0.5,0.5,0.5), lw=0.5)

# Draw arrows
vec = np.repeat(np.array([[0,0,-1.0]]), positions.shape[0], axis=0)
for i in range(vec.shape[0]):
    vec[i,:] = rotate_by_quaternion(vec[i,:], orientations[i,:])
arrows = np.concatenate((positions, vec), axis=1).T
X,Y,Z,U,V,W = arrows
ax.quiver(X,Y,Z,U,V,W, pivot='tail', color='red', length=args.arrow_len, lw=1)

if args.connect:
    ax.plot(positions[:,0], positions[:,1], positions[:,2], color='red')

if args.plot_gt:
    ax.plot(positions_gt[:,0], positions_gt[:,1], positions_gt[:,2], color='black')

if args.plot_diff:
    for i in range(positions.shape[0]): # Yeah I know it's slow
        px, py, pz = positions_gt[i][:]
        qx, qy, qz = positions[i][:]
        plt.plot([px, qx], [py, qy], [pz, qz], linestyle='-', color=(0.5,0.5,0.5))


# Plot limits
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