import tensorflow as tf

from tensorflow.contrib.layers import initializers
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim.nets import inception
import numpy as np

def create_image_summary(V, size, name):
    if type(V) == str:
        V = tf.get_default_graph().get_tensor_by_name(V)

    V = tf.slice(V,(0,0,0,0),(1,-1,-1,-1))
    batches, ix, iy, channels = V.get_shape().as_list()
    V = tf.reshape(V, (ix, iy, channels))

    # Add some padding
    ix += 4
    iy += 4
    V = tf.image.resize_image_with_crop_or_pad(V, iy, ix)

    cx = size[1]
    cy = size[0]
    V = tf.reshape(V,(iy,ix,cy,cx))
    V = tf.transpose(V,(2,0,3,1)) #cy,iy,cx,ix
    V = tf.reshape(V,(1,cy*iy,cx*ix,1))

    return tf.image_summary(name, V)

def create_histogram_summary(tensor_name):
    H = tf.get_default_graph().get_tensor_by_name("PoseNet/" + tensor_name + ":0")
    return tf.histogram_summary(tensor_name, H)

class Posenet:

    # Building blocks
    weight_decay = staticmethod(slim.l2_regularizer(5e-5))
    activation_function = staticmethod(tf.nn.relu)
    normalizer_function = staticmethod(slim.batch_norm)

    # Use the "MSRA" initialization
    weight_init = staticmethod(initializers.variance_scaling_initializer())

    # batch norm settings
    batch_norm_decay = 0.9997
    batch_norm_epsilon = 0.001
    batch_norm_params = {
        'decay': batch_norm_decay,
        'epsilon': batch_norm_epsilon,
    }

    def __init__(self):
        pass

    def create_stream(self, data_input, dropout, trainable):

        self.batch_norm_params['trainable'] = trainable

        with slim.arg_scope([slim.conv2d], padding='SAME',
                            activation_fn=self.activation_function, weights_initializer=self.weight_init,
                            weights_regularizer=self.weight_decay, #normalizer_fn=self.normalizer_function,
                            normalizer_params=self.batch_norm_params, trainable=trainable):

            with slim.arg_scope([slim.batch_norm], **self.batch_norm_params):
                last_output, layers = inception.inception_v3_base(data_input, scope='Inception_V3',
                                                          final_endpoint='Mixed_7c')
                # Global average pooling
                last_output = tf.reduce_mean(last_output, [1, 2])
                last_output = tf.reshape(last_output, [-1, 2048])
                last_output = slim.fully_connected(last_output, 2048, scope='fc0', trainable=trainable)
            if dropout is not None:
                last_output = slim.dropout(last_output, keep_prob=dropout, scope='dropout_0', is_training=trainable)

        # Pose Regression
        last_output = slim.fully_connected(last_output, 7,
                                            normalizer_fn=None, scope='fc1',
                                            activation_fn=None, weights_initializer=self.weight_init,
                                            weights_regularizer=self.weight_decay, trainable=trainable)
        
        return self.slice_output(last_output), layers

    def slice_output(self, output):
        x = tf.slice(output, [0, 0], [-1, 3])
        q = tf.nn.l2_normalize(tf.slice(output, [0, 3], [-1, 4]), 1)
        return {'x': x, 'q': q}

    def loss(self, outputs, gt, beta):
        x_loss = tf.reduce_sum(tf.abs(tf.sub(outputs["x"], gt["x"])), 1)
        q_loss = tf.reduce_sum(tf.abs(tf.sub(outputs["q"], gt["q"])), 1)
        #x_loss = tf.sqrt(tf.reduce_sum(tf.square(tf.sub(outputs["x"], gt["x"])), 1) + 1e-10)
        #q_loss = tf.acos(tf.clip_by_value((2*tf.square(tf.reduce_sum(tf.mul(outputs["q"], gt["q"]), 1)) - 1.0), -1.0, 1.0))

        x_loss = tf.reduce_mean(x_loss)
        q_loss = tf.reduce_mean(q_loss)
        total_loss = tf.add(x_loss, tf.mul(q_loss, tf.constant(beta, tf.float32)))

        return x_loss, q_loss, total_loss

    def create_validation(self, inputs, labels, beta=500, reuse=False):
        summaries = []
        with tf.variable_scope('PoseNet', reuse=reuse):
            outputs, layers = self.create_stream(inputs, dropout=None, trainable=False)
            gt = self.slice_output(labels)
            x_loss, q_loss, total_loss = self.loss(outputs, gt, beta)

            # And scalar smmaries
            summaries.append(tf.scalar_summary('validation/Positional Loss', x_loss))
            summaries.append(tf.scalar_summary('validation/Orientation Loss', q_loss))
            summaries.append(tf.scalar_summary('validation/Total Loss', total_loss))

        return outputs, total_loss, summaries

    def create_testable(self, inputs):
        with tf.variable_scope('PoseNet'):
            outputs, _ = self.create_stream(inputs, dropout=None, trainable=False)
        return outputs

    def create_trainable(self, inputs, labels, dropout=0.7, beta=500):
        summaries = []
        with tf.variable_scope('PoseNet'):

            outputs, layers = self.create_stream(inputs, dropout, trainable=True)
            gt = self.slice_output(labels)
            x_loss, q_loss, total_loss = self.loss(outputs, gt, beta)

            # Create some image summaries
            #create_image_summary("PoseNet/Inception_V3/Conv2d_1a_3x3/weights:0", [8, 4], "Conv2d_1a_3x3/weights")
            summaries.append(create_image_summary(layers["Conv2d_1a_3x3"], [8, 4], "Conv2d_1a_3x3"))
            summaries.append(create_image_summary(layers["Conv2d_2a_3x3"], [8, 4], "Conv2d_2a_3x3"))
            summaries.append(create_image_summary(layers["Conv2d_2b_3x3"], [16, 4], "Conv2d_2b_3x3"))
            summaries.append(create_image_summary(layers["Conv2d_3b_1x1"], [16, 5], "Conv2d_3b_1x1"))
            summaries.append(create_image_summary(layers["Conv2d_4a_3x3"], [32, 6], "Conv2d_4a_3x3"))

            # And scalar smmaries
            summaries.append(tf.scalar_summary('train/Positional Loss', x_loss))
            summaries.append(tf.scalar_summary('train/Orientation Loss', q_loss))
            summaries.append(tf.scalar_summary('train/Total Loss', total_loss))

            # And histogram summaries
            summaries.append(create_histogram_summary("fc0/weights"))
            summaries.append(create_histogram_summary("fc0/biases"))
            summaries.append(create_histogram_summary("fc1/weights"))
            summaries.append(create_histogram_summary("fc1/biases"))

        #print map(lambda x: x.name, tf.get_collection(tf.GraphKeys.VARIABLES, scope='PoseNet'))
        #with tf.variable_scope("PoseNet"):
        #    print tf.get_variable("Inception_V3/Mixed_5c/Branch_1/Conv_1_0c_5x5/weights:0")

        return outputs, total_loss, summaries