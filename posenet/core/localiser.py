import numpy
import tensorflow as tf

import numpy as np

from .posenet import Posenet
import matplotlib.pyplot as plt


class Localiser:
    def __init__(self, model_path, uncertainty=False, output_type='quat', dropout=0.5):
        # Define the network
        self.x = tf.placeholder(tf.float32, [None, None, None, 3], name="InputData")
        self.network = Posenet(endpoint='Mixed_5b', n_fc=256, output_type=output_type)
        self.uncertainty = uncertainty
        if uncertainty:
            self.output = self.network.create_testable(self.x, dropout=dropout)
        else:
            self.output = self.network.create_testable(self.x, dropout=None)
        self.model_path = model_path

        # Initialise other stuff
        self.saver = tf.train.Saver()
        self.init = tf.global_variables_initializer()
        self.session = None

    def __enter__(self):
        self.session = tf.Session()
        self.session.run(self.init)
        self.saver.restore(self.session, self.model_path) # Load the model
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.session.close()

    def _localise(self, img):
        predicted = self.session.run([self.output], feed_dict={self.x: img})
        return {'x': predicted[0]['x'], 'q': predicted[0]['q']}

    def localise(self, img, samples=10):
        if len(img.shape) == 3:
            img = np.expand_dims(img, axis=0)

        if self.uncertainty:
            pred = self._localise(np.repeat(img, samples, axis=0))

            x = list(np.mean(pred['x'], axis=0))
            q = list(np.mean(pred['q'], axis=0))
            std_x_all = np.std(pred['x'], axis=0)
            std_q_all = np.std(pred['q'], axis=0)
            std_x = sum(std_x_all)
            std_q = sum(std_q_all)

            return {'x': x, 'q': q, 'std_x': std_x, 'std_q': std_q, 
                    'std_x_all': std_x_all, 'std_q_all': std_q_all}
        else:
            pred = self._localise(img)
            return {'x': pred['x'][0], 'q': pred['q'][0]}

    def saliency(self, img):
        if len(img.shape) == 3:
            img = np.expand_dims(img, axis=0)

        layer = self.network.layers['last_output']

        t_obj = tf.reduce_mean(layer[0])
        t_grad = tf.gradients(t_obj, self.x)[0][0,:,:,:]

        grad = self.session.run(t_grad, {self.x: img})

        grad = np.max(np.abs(grad), axis=2)
        grad = grad/np.max(grad)

        return grad
