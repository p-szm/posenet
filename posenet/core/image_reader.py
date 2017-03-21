import math
import os
import random

import numpy as np
from skimage import io, transform, exposure, img_as_float
from skimage.util import random_noise
from skimage.filters import gaussian

from ..utils import quat_to_axis


def read_label_file(def_file, full_paths=False, convert_to_axis=False):
    paths = []
    labels = []
    with open(def_file) as f:
        lines = map(lambda line: line.rstrip("\n").split(" "), f.readlines()[3:])
        paths = map(lambda line: line[0], lines)
        labels = map(lambda line: map(lambda x: float(x), line[1:]), lines)
    if full_paths:
        paths = map(lambda x: os.path.join(os.path.dirname(def_file), x), paths)
    if convert_to_axis:
        labels = map(lambda l: l[0:3] + list(quat_to_axis(l[3:7])), labels) 
    return list(paths), list(labels)


def read_image(path, size=None, expand_dims=False, normalise=False):
    image = img_as_float(io.imread(path))
    if size is not None:
        image = transform.resize(image, (size[0], size[1], 3))
    if normalise:
        image = 2 * image - 1
    if expand_dims:
        image = np.expand_dims(image, axis=0)
    return image


class ImageReader:
    def __init__(self, def_file, batch_size=1, image_size=[224,224],
                 crop_size=None, centre_crop=False, randomise=False,
                 augment=False, convert_to_axis=False):
        self.image_dir = os.path.dirname(def_file)
        self.batch_size = batch_size
        self.image_size = image_size
        self.crop_size = crop_size
        self.centre_crop = centre_crop
        self.randomise = randomise
        self.augment = augment
        self.images, self.labels = read_label_file(def_file, convert_to_axis=convert_to_axis)
        self.idx = 0
        if randomise:
            self._shuffle()

    def _full_path(self, f):
        return os.path.join(self.image_dir, f)

    def _shuffle(self):
        index_shuf = range(len(self.images))
        random.shuffle(index_shuf)
        self.images = [self.images[i] for i in index_shuf]
        self.labels = [self.labels[i] for i in index_shuf]

    def _reset(self):
        self.idx = 0
        if self.randomise:
            self._shuffle()

    def _augment(self, image, gamma_range=(0.33, 3.0), 
                 gauss_range = (0, 4), noise_range = (0, 0.02),
                 color_range = (0.5, 1.5), chance=1):
        if random.uniform(0, 1) < chance:
            # Colorize the image
            image = np.random.uniform(color_range[0], color_range[1], [1,3]) * image
            image = np.clip(image, -1.0, 1.0)
        if random.uniform(0, 1) < chance:
            image = exposure.adjust_gamma(image, random.uniform(*gamma_range))
        if random.uniform(0, 1) < chance:
            image = gaussian(image, random.uniform(*gauss_range), multichannel=True)
        if random.uniform(0, 1) < chance:
            image = random_noise(image, mode='gaussian', seed=None, clip=True, 
                var=random.uniform(*noise_range))
        return image

    def _read_image(self, image_path):
        image = read_image(self._full_path(image_path), size=self.image_size, normalise=False)

        if self.augment:
            image = self._augment(image, noise_range=(0, 0.005), chance=0.5)
        if self.crop_size is not None:
            if self.centre_crop:
                offset_x = int(math.floor((self.image_size[0]-self.crop_size[0])/2))
                offset_y = int(math.floor((self.image_size[1]-self.crop_size[1])/2))
            else:
                offset_x = random.randint(0, self.image_size[0]-self.crop_size[0])
                offset_y = random.randint(0, self.image_size[1]-self.crop_size[1])
            image = image[offset_y:offset_y+self.crop_size[1],
                          offset_x:offset_x+self.crop_size[0], :]

        # Normalise to [-1,1]
        image = 2 * image - 1
        return image

    def next_batch(self):
        images = list(map(lambda image: self._read_image(image), 
                    self.images[self.idx:self.idx+self.batch_size]))
        images = np.asarray(images)
        labels = self.labels[self.idx:self.idx+self.batch_size]
        self.idx += self.batch_size
        if self.idx >= len(self.images):
            self._reset()
        return images, labels

    def total_images(self):
        return len(self.images)

    def total_batches(self):
        return int(math.ceil(len(self.images)/self.batch_size))
