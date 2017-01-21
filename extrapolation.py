import argparse

import numpy as np

from posenet.core.image_reader import ImageReader
from posenet.core.localiser import Localiser
from posenet.utils import l2_distance, quaternion_distance, rotate_by_quaternion


parser = argparse.ArgumentParser(description='''
    Tests a model that was trained on a single object from multiple sides''')
parser.add_argument('--model', action='store', required=True, 
    help='''Path to a trained Tensorflow model (.ckpt file)''')
parser.add_argument('--dataset', action='store', required=True, 
    help='''Path to a text file listing images and camera poses''')
parser.add_argument('--plot_errors', action='store', nargs='?', const='',
    help='''Plot positional and orientation errors vs azimuthal angle 
    (from -90 to 90 deg). If no filename is specified it is plotted 
    interactively.''')
parser.add_argument('--plot_3d', action='store', nargs='?', const='', 
    help='''Produce a 3d plot of predicted camera poses. If no filename 
    is specified it is plotted interactively''')
parser.add_argument('--trace_path', action='store_true',
    help='''For use with --plot_3d argument. Connect consecutive camera 
    positions with a line''')
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
    shade = (1, 0.8, 0.8)
    ax1.axvspan(-90, -45, color=shade)
    ax1.axvspan(45, 90, color=shade)
    ax1.set_ylabel("Positional error")

    ax2.plot(x, orient_errors, color='black')
    ax2.set_xlim([-90, 90])
    ax2.axvspan(-90, -45, color=shade)
    ax2.axvspan(45, 90, color=shade)
    ax2.set_xlabel('Phi')
    ax2.set_ylabel("Orientation error (degrees)")

    if args.plot_errors:
        plt.savefig(args.plot_errors, bbox_inches='tight')
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
    ax.plot_wireframe(x, y, z, rstride=4, cstride=4, color='black', lw=0.5)

    if args.trace_path:
        # Correct path
        phi = np.linspace(-np.pi/2, np.pi/2, 100)
        ax.plot(8*np.cos(phi), 8*np.sin(phi), np.zeros(100), color='black', lw=0.5)

        # Path taken
        ax.plot(positions[:,0], positions[:,1], positions[:,2], color='red')

    # Arrows
    vec = np.repeat(np.array([[0,0,-1.0]]), positions.shape[0], axis=0)
    for i in xrange(vec.shape[0]):
        vec[i,:] = rotate_by_quaternion(vec[i,:], orientations[i,:])
    arrows = np.concatenate((positions, vec), axis=1).T
    X,Y,Z,U,V,W = arrows
    ax.quiver(X,Y,Z,U,V,W, pivot='tail', color='blue', length=1.5, lw=0.5)

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

    if args.plot_3d:
        plt.savefig(args.plot_3d, bbox_inches='tight')
    else:
        plt.show()