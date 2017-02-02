import numpy
import tensorflow as tf

import numpy as np

from .posenet import Posenet


class Localiser:
    def __init__(self, input_size, model_path, uncertainty=False):
        # Define the network
        self.x = tf.placeholder(tf.float32, [None, input_size, input_size, 3], name="InputData")
        self.network = Posenet(endpoint='Mixed_5b', n_fc=256)
        self.uncertainty = uncertainty
        if uncertainty:
            self.output = self.network.create_testable(self.x, dropout=0.5)
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
        return {'x': predicted[0]['x'][0].tolist(), 'q': predicted[0]['q'][0].tolist()}

    def localise(self, img):
        """Accepts a numpy image [size, size, n_channels] or [1, size, size, n_channels]"""
        if len(img.shape) == 3:
            img = np.expand_dims(img, axis=0)

        if self.uncertainty:
            positions = np.empty([0,3])
            orientations = np.empty([0,4])

            for i in range(10):
                pred = self._localise(img)
                positions = np.concatenate((positions, np.asarray([pred['x']])))
                orientations = np.concatenate((orientations, np.asarray([pred['q']])))
            
            x = list(np.mean(positions, 0))
            q = list(np.mean(orientations, 0))
            std_x = sum(list(np.std(positions, 0)))
            std_q = sum(list(np.std(orientations, 0)))
            return {'x': x, 'q': q, 'std_x': std_x, 'std_q': std_q}
        else:
            return self._localise(img)
            pred.x
