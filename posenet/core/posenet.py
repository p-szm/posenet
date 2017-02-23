import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.layers import variance_scaling_initializer
from tensorflow.contrib.slim.nets import inception


class Posenet:

    # Building blocks
    weight_decay = staticmethod(slim.l2_regularizer(5e-5))
    activation_function = staticmethod(tf.nn.relu)
    normalizer_function = staticmethod(slim.batch_norm)

    # Use the "MSRA" initialization
    weight_init = staticmethod(variance_scaling_initializer())

    # batch norm settings
    batch_norm_decay = 0.9997
    batch_norm_epsilon = 0.001
    batch_norm_params = {
        'decay': batch_norm_decay,
        'epsilon': batch_norm_epsilon,
    }

    def __init__(self, endpoint='Mixed_7c', n_fc=2048):
        self.endpoint = endpoint
        self.n_fc = n_fc
        self.layers = {}

    def create_stream(self, data_input, dropout, trainable):

        self.batch_norm_params['trainable'] = trainable

        with slim.arg_scope([slim.conv2d], padding='SAME',
                            activation_fn=self.activation_function, weights_initializer=self.weight_init,
                            weights_regularizer=self.weight_decay, #normalizer_fn=self.normalizer_function,
                            normalizer_params=self.batch_norm_params, trainable=trainable):

            with slim.arg_scope([slim.batch_norm], **self.batch_norm_params):
                last_output, layers = inception.inception_v3_base(data_input, scope='Inception_V3',
                                                          final_endpoint=self.endpoint)
                # Global average pooling
                last_output = tf.reduce_mean(last_output, [1, 2])
                last_output = tf.reshape(last_output, [-1, self.n_fc])
                last_output = slim.fully_connected(last_output, self.n_fc, scope='fc0', trainable=trainable)
            if dropout is not None:
                last_output = slim.dropout(last_output, keep_prob=dropout, scope='dropout_0')

        # Pose Regression
        last_output = slim.fully_connected(last_output, 7,
                                            normalizer_fn=None, scope='fc1',
                                            activation_fn=None, weights_initializer=self.weight_init,
                                            weights_regularizer=self.weight_decay, trainable=trainable)

        layers['last_output'] = last_output
        self.layers = layers
        return self.slice_output(last_output), layers

    def slice_output(self, output):
        x = tf.slice(output, [0, 0], [-1, 3])
        q = tf.nn.l2_normalize(tf.slice(output, [0, 3], [-1, 4]), 1)
        return {'x': x, 'q': q}

    def loss(self, outputs, gt, beta, learn_beta):
        x_loss = tf.reduce_sum(tf.abs(tf.subtract(outputs["x"], gt["x"])), 1)
        q_loss = tf.reduce_sum(tf.abs(tf.subtract(outputs["q"], gt["q"])), 1)
        #x_loss = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(outputs["x"], gt["x"])), 1) + 1e-10)
        #q_loss = tf.acos(tf.clip_by_value((2*tf.square(tf.reduce_sum(tf.multiply(outputs["q"], gt["q"]), 1)) - 1.0), -1.0, 1.0))

        x_loss = tf.reduce_mean(x_loss)
        q_loss = tf.reduce_mean(q_loss)

        if learn_beta:
            total_loss = tf.add(tf.truediv(x_loss, beta), tf.multiply(q_loss, beta))
        else:
            total_loss = tf.add(x_loss, tf.multiply(q_loss, beta))

        return x_loss, q_loss, total_loss

    def create_validation(self, inputs, labels, beta=None):
        summaries = []
        with tf.variable_scope('PoseNet', reuse=True):
            try:
                weight = tf.get_default_graph().get_tensor_by_name('PoseNet/learned_beta:0')
                learn_beta = True
                print('Using learned beta')
            except KeyError:
                if beta is None:
                    raise ValueError('The value of beta has to be specified')
                weight = tf.constant(beta, tf.float32)
                learn_beta = False

            outputs, layers = self.create_stream(inputs, dropout=None, trainable=False)
            gt = self.slice_output(labels)
            x_loss, q_loss, total_loss = self.loss(outputs, gt, weight, learn_beta)

            # And scalar smmaries
            summaries.append(tf.summary.scalar('validation/Positional Loss', x_loss))
            summaries.append(tf.summary.scalar('validation/Orientation Loss', q_loss))
            summaries.append(tf.summary.scalar('validation/Total Loss', total_loss))

        return outputs, total_loss, summaries

    def create_testable(self, inputs, dropout=None):
        with tf.variable_scope('PoseNet'):
            outputs, _ = self.create_stream(inputs, dropout=dropout, trainable=False)
        return outputs

    def create_trainable(self, inputs, labels, dropout=0.7, beta=500, learn_beta=False):
        summaries = []
        with tf.variable_scope('PoseNet'):

            outputs, layers = self.create_stream(inputs, dropout, trainable=True)
            gt = self.slice_output(labels)

            if learn_beta:
                weight = tf.Variable(tf.constant(beta, tf.float32), name="learned_beta")
                summaries.append(tf.summary.scalar('train/Beta', weight))
            else:
                weight = tf.constant(beta, tf.float32)

            x_loss, q_loss, total_loss = self.loss(outputs, gt, weight, learn_beta)

            # And scalar smmaries
            summaries.append(tf.summary.scalar('train/Positional Loss', x_loss))
            summaries.append(tf.summary.scalar('train/Orientation Loss', q_loss))
            summaries.append(tf.summary.scalar('train/Total Loss', total_loss))

        return outputs, total_loss, summaries
