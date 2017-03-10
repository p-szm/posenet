import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from posenet.core.image_reader import *
from posenet.core.localiser import Localiser
from posenet.utils import progress_bar


def plot_sigma(angle, y, sigma):
    # (68-95-99.7 rule)
    plt.fill_between(angle, y-sigma, y+sigma, 
        facecolor=(0,0,0,0.1), linewidth=0.0)

def plot_verticals(x):
    for xc in x:
        plt.axvline(x=xc, color='k', linestyle=':', lw=0.5)


mode = 'interpolation'
dataset = 'datasets/david/test1.txt'
model = 'models/david_dropout/david_dropout_iter9000.ckpt'
input_size = 256
k = 10
x_min, x_max = 0, 2*np.pi
shade = (0, 0.6, 0, 0.05)


img_paths, labels = read_label_file(dataset, full_paths=True, convert_to_axis=True)
n_images = len(img_paths)

x_scale = np.linspace(x_min, x_max, n_images)
x_gt = map(lambda l: l[0:3], labels)
q_gt = map(lambda l: l[3:], labels)
x = []
q = []
std_x = []
std_q = []

with Localiser(model, uncertainty=True, output_type='axis', dropout=0.3) as localiser:
    for i, path in enumerate(img_paths):
        img = read_image(path, size=[input_size, input_size], 
                expand_dims=False, normalise=True)
        pred = localiser.localise(img, samples=40)
        x.append(pred['x'])
        q.append(pred['q'])
        std_x.append(pred['std_x_all'])
        std_q.append(pred['std_q_all'])
        progress_bar(1.0*(i+1)/n_images, 30, text='Localising')
print ''

x_gt = np.array(x_gt)
q_gt = np.array(q_gt)
x = np.array(x)
q = np.array(q)
std_x = np.array(std_x)
std_q = np.array(std_q)

for i in range(3):
    plot_sigma(x_scale, x[:,i], 2*std_x[:,i])
    plt.plot(x_scale, x_gt[:,i], color='black', lw=0.5)
    plt.plot(x_scale, x[:,i])
    if mode == 'interpolation':
        plt.plot(x_scale[::k], x_gt[::k,i], marker='.', lw=0, color='black')
    else:
        plt.axvspan(np.pi/2, np.pi*5/6, color=shade)
        plt.axvspan(np.pi*7/6, np.pi*3/2, color=shade)
    plt.xlim([x_min, x_max])
    #plt.ylim([1.6*np.min(x_gt), 1.6*np.max(x_gt)])
if mode == 'interpolation':
    plot_verticals(x_scale[::k])
plt.savefig('plots/david_dropout/x_9k.svg', bbox_inches='tight')

plt.figure()
for i in range(3):
    plot_sigma(x_scale, q[:,i], 2*std_q[:,i])
    plt.plot(x_scale, q_gt[:,i], color='black', lw=0.5)
    plt.plot(x_scale, q[:,i])
    if mode == 'interpolation':
        plt.plot(x_scale[::k], q_gt[::k,i], marker='.', lw=0, color='black')
    else:
        plt.axvspan(np.pi/2, np.pi*5/6, color=shade)
        plt.axvspan(np.pi*7/6, np.pi*3/2, color=shade)
    plt.xlim([x_min, x_max])
    #plt.ylim([1.2*np.min(q_gt), 1.2*np.max(q_gt)])
if mode == 'interpolation':
    plot_verticals(x_scale[::k])
plt.savefig('plots/david_dropout/q_9k.svg', bbox_inches='tight')
