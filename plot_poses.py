import argparse

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

from posenet.core.image_reader import read_label_file
from posenet.utils import rotate_by_quaternion


parser = argparse.ArgumentParser(description='''
    Visualise camera poses given a definition file''')
parser.add_argument('--dataset', action='store', required=True, 
    help='''Path to a text file listing images and camera poses''')
args = parser.parse_args()


_, labels = read_label_file(args.dataset)
positions = np.array([l[0:3] for l in labels])
orientations = np.array([l[3:7] for l in labels])

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
plt.show()