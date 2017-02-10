import argparse
from time import sleep

import matplotlib.pyplot as plt

from posenet.core.image_reader import ImageReader


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', action='store', required=True)
args = parser.parse_args()


input_size = 224
crop_size = 200
reader = ImageReader(args.dataset, batch_size=16, 
                    image_size=[input_size, input_size], 
                    randomise=True, augment=True, crop_size=[crop_size, crop_size])

fig = plt.figure(num=None, figsize=(6, 6), dpi=80, facecolor='w')
image, label = reader.next_batch()
for j in range(16):
    fig.add_subplot(4, 4, j+1)
    plt.imshow((1+image[j])/2.0, aspect='auto')
    plt.axis('off')
plt.subplots_adjust(hspace=0, wspace=0)
plt.show()