import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import sys

from posenet.core.image_reader import *
from posenet.core.localiser import Localiser
from posenet.utils import progress_bar
from posenet.utils import quat_to_axis


def plot_sigma(angle, y, sigma):
    # (68-95-99.7 rule)
    plt.fill_between(angle, y-sigma, y+sigma, 
        facecolor=(0,0,0,0.1), linewidth=0.0)

def plot_verticals(x):
    for xc in x:
        plt.axvline(x=xc, color='k', lw=0.3)

def subtract_trend(positions, sigmas):

    p_all, s_all = positions.flatten(), sigmas.flatten()
    p_all, s_all = zip(*sorted(zip(np.abs(p_all), s_all)))
    p_all, s_all = np.array(p_all), np.array(s_all)

    # Use only samples for which the uncertainty is significant
    # (some of the samples have very high confidence which screws
    # up the fit)
    p, s = p_all[s_all > 0.001], s_all[s_all > 0.001]
    
    a, b = np.polyfit(p, s, 1)

    #plt.plot(p_all, s_all)
    #plt.plot(p, p*a+b)
    #plt.savefig('results_new/david/a.pdf')

    s_corrected = sigmas - np.abs(positions) * a
    s_corrected[s_corrected < 0.0] = 0.0

    return s_corrected


mode = 'interpolation'
dataset = 'datasets/david/test1.txt'
model = 'models_new/david_pretrained/david_pretrained_iter12000.ckpt'
errors = False
input_size = 256
shade = (0.8, 0.8, 0.8, 0.5)

k = 10
x_min, x_max = 0, 2*np.pi
#k = 20
#x_min, x_max = 0.5, 5.997787


img_paths, labels = read_label_file(dataset, full_paths=True, convert_to_axis=True)
n_images = len(img_paths)

x_scale = np.linspace(x_min, x_max, n_images)
x_gt = map(lambda l: l[0:3], labels)
q_gt = map(lambda l: l[3:], labels)
x = []
q = []
std_x = []
std_q = []

with Localiser(model, uncertainty=False, output_type='axis', dropout=0.5) as localiser:
    for i, path in enumerate(img_paths):
        img = read_image(path, expand_dims=False, normalise=True)
        img = resize_image(centre_crop(img), [input_size, input_size])
        pred = localiser.localise(img, samples=40)
        x.append(pred['x'])
        q.append(pred['q'])
        if 'std_x_all' in pred:
            std_x.append(pred['std_x_all'])
            std_q.append(pred['std_q_all'])
        progress_bar(1.0*(i+1)/n_images, 30, text='Localising')
print ''

x_gt = np.array(x_gt)
q_gt = np.array(q_gt)
x = np.array(x)
q = np.array(q)

if std_x:
    std_x = np.array(std_x)
    std_q = np.array(std_q)
    std_x = subtract_trend(x, std_x)
    std_q = subtract_trend(q, std_q)

if errors:
    errs_x = np.linalg.norm(x-x_gt, ord=2, axis=1)
    errs_q = np.linalg.norm(q-q_gt, ord=2, axis=1)
    errs_angle = 180.0/np.pi*np.arccos(np.sum(q*q_gt, axis=1))
    #errs_angle = []
    #for i in range(q.shape[0]):
    #    q1 = quat_to_axis(q[i,:])
    #    q2 = quat_to_axis(q_gt[i,:])
    #    errs_angle.append(180.0/np.pi*np.arccos(np.sum(q1*q2)))
    #errs_angle = np.array(errs_angle)
    print 'l2 x: ', round(np.mean(errs_x), 2)
    print '90 perc: ', round(np.percentile(errs_x, 90), 2)
    print '95 perc: ', round(np.percentile(errs_x, 95), 2)
    print '99 perc: ', round(np.percentile(errs_x, 99), 2)
    print ''
    print 'l2 q: ', round(np.mean(errs_q), 2)
    print '90 perc: ', round(np.percentile(errs_q, 90), 2)
    print '95 perc: ', round(np.percentile(errs_q, 95), 2)
    print '99 perc: ', round(np.percentile(errs_q, 99), 2)
    print ''
    print 'Angle: ', round(np.mean(errs_angle), 2)
    print '90 perc: ', round(np.percentile(errs_angle, 90), 2)
    print '95 perc: ', round(np.percentile(errs_angle, 95), 2)
    print '99 perc: ', round(np.percentile(errs_angle, 99), 2)
    sys.exit(0)

colors = ['r', 'g', 'b']
labels = ['x', 'y', 'z']
figsize = (4,3.5)

plt.figure(figsize=figsize)
for i in range(3):
    if std_x:
        plot_sigma(x_scale, x[:,i], 2*std_x[:,i])
    plt.plot(x_scale, x_gt[:,i], color='black', lw=0.5)
    plt.plot(x_scale, x[:,i], color=colors[i], label=labels[i], lw=0.9)
    #if mode == 'interpolation':
    #    plt.plot(x_scale[::k], x_gt[::k,i], marker='.', lw=0, color='black')
    if mode == 'extrapolation':
        plt.axvspan(np.pi/2, np.pi*5/6, color=shade)
        plt.axvspan(np.pi*7/6, np.pi*3/2, color=shade)
    plt.xlim([x_min, x_max])
    plt.ylim([1.1*np.min(x), 1.1*np.max(x)])
    plt.xlabel('Angular position')
    plt.ylabel('Position vector')
if mode == 'interpolation':
    plot_verticals(x_scale[::k])
plt.legend(loc='lower left')
plt.subplots_adjust(bottom=0.15,left=0.18)
#ax = plt.axes()
#ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('$%g\pi$'))
#ax.xaxis.set_major_locator(ticker.MultipleLocator(base=0.5))
plt.savefig('results_new/david_pretrained/x_test1_12k.pdf')

plt.figure(figsize=figsize)
for i in range(3):
    if std_q:
        plot_sigma(x_scale, q[:,i], 2*std_q[:,i])
    plt.plot(x_scale, q_gt[:,i], color='black', lw=0.5)
    plt.plot(x_scale, q[:,i], color=colors[i], label=labels[i], lw=0.9)
    #if mode == 'interpolation':
    #    plt.plot(x_scale[::k], q_gt[::k,i], marker='.', lw=0, color='black')
    if mode == 'extrapolation':
        plt.axvspan(np.pi/2, np.pi*5/6, color=shade)
        plt.axvspan(np.pi*7/6, np.pi*3/2, color=shade)
    plt.xlim([x_min, x_max])
    #plt.ylim([1.2*np.min(q_gt), 1.2*np.max(q_gt)])
    plt.xlabel('Angular position')
    plt.ylabel('Orientation vector')
if mode == 'interpolation':
    plot_verticals(x_scale[::k])
plt.legend(loc='upper left')
plt.subplots_adjust(bottom=0.15, left=0.18)
#ax = plt.axes()
#ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('$%g\pi$'))
#ax.xaxis.set_major_locator(ticker.MultipleLocator(base=0.5))
plt.savefig('results_new/david_pretrained/q_test1_12k.pdf')
