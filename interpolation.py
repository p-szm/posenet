import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from posenet.core.image_reader import *
from posenet.core.localiser import Localiser
from posenet.utils import progress_bar


def plot_sigma(angle, y, sigma):
    # (68-95-99.7 rule)
    plt.fill_between(angle, y-2*sigma, y+2*sigma, 
        facecolor=(0.9,0.9,0.9), linewidth=0.0)

def plot_verticals(x):
    for xc in x:
        plt.axvline(x=xc, color='k', linestyle=':')


dataset = 'datasets/david/test1.txt'
model = 'models/david/david_iter18000.ckpt'
input_size = 256

img_paths, labels = read_label_file(dataset, full_paths=True, convert_to_axis=True)
n_images = len(img_paths)

angle = np.linspace(-np.pi, np.pi, n_images)
x_gt = map(lambda l: l[0:3], labels)
q_gt = map(lambda l: l[3:], labels)
x = []
q = []
std_x = []
std_q = []

with Localiser(model, uncertainty=True, output_type='axis') as localiser:
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
    plot_sigma(angle, x[:,i], std_x[:,i])
    plt.plot(angle, x_gt[:,i], color='black')
    plt.plot(angle, x[:,i])
    plt.plot(angle[::10], x_gt[::10,i], marker='.', lw=0, color='black')
    plt.xlim([min(angle), max(angle)])
    plt.ylim([1.6*np.min(x_gt), 1.6*np.max(x_gt)])
plot_verticals(angle[::10])
plt.savefig('plots/david/x_18k.eps', bbox_inches='tight')

plt.figure()
for i in range(3):
    plot_sigma(angle, q[:,i], std_q[:,i])
    plt.plot(angle, q_gt[:,i], color='black')
    plt.plot(angle, q[:,i])
    plt.plot(angle[::10], q_gt[::10,i], marker='.', lw=0, color='black')
    plt.xlim([min(angle), max(angle)])
    plt.ylim([1.2*np.min(q_gt), 1.2*np.max(q_gt)])
plot_verticals(angle[::10])
plt.savefig('plots/david/q_18k.eps', bbox_inches='tight')