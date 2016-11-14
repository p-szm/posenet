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
    tf.image_summary(name, V)


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
                last_output, _ = inception.inception_v3_base(data_input, scope='Inception_V3',
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
        x = tf.slice(last_output, [0, 0], [-1, 3])
        q = tf.slice(last_output, [0, 3], [-1, 4])

        # normalise vectors to unit length
        q = tf.nn.l2_normalize(q, 1)
        
        output = {'x': x, 'q': q}
        return output, _

    def create_testable(self, inputs, reuse=False):
        with tf.variable_scope('PoseNet', reuse=reuse):
            return self.create_stream(inputs, dropout=None, trainable=False)

    def create_trainable(self, inputs, labels, dropout=0.7, beta=500):
        with tf.variable_scope('PoseNet'):
            outputs, layers = self.create_stream(inputs, dropout, trainable=True)

            # separate pose label
            x_gt = tf.slice(labels, [0, 0], [-1, 3])
            q_gt = tf.slice(labels, [0, 3], [-1, 4])

            x_loss = tf.sqrt(tf.reduce_sum(tf.square(tf.sub(outputs["x"], x_gt)), 1) + 1e-10)
            q_loss = tf.acos(tf.clip_by_value((2*tf.square(tf.reduce_sum(tf.mul(outputs["q"], q_gt), 1)) - 1.0), -1.0, 1.0))

            x_loss = tf.reduce_mean(x_loss)
            q_loss = tf.reduce_mean(q_loss)

            total_loss = tf.add(x_loss, tf.mul(q_loss, tf.constant(beta, tf.float32)))

            # Create some image summaries
            #create_image_summary("PoseNet/Inception_V3/Conv2d_1a_3x3/weights:0", [8, 4], "Conv2d_1a_3x3/weights")
            create_image_summary(layers["Conv2d_1a_3x3"], [8, 4], "Conv2d_1a_3x3")
            create_image_summary(layers["Conv2d_2a_3x3"], [8, 4], "Conv2d_2a_3x3")
            create_image_summary(layers["Conv2d_2b_3x3"], [16, 4], "Conv2d_2b_3x3")
            create_image_summary(layers["Conv2d_3b_1x1"], [16, 5], "Conv2d_3b_1x1")
            create_image_summary(layers["Conv2d_4a_3x3"], [32, 6], "Conv2d_4a_3x3")

            # And scalar smmaries
            tf.scalar_summary('Positional Loss', x_loss)
            tf.scalar_summary('Orientation Loss', q_loss)
            tf.scalar_summary('Total Loss', total_loss)

            # And histogram summaries
            H = tf.get_default_graph().get_tensor_by_name("PoseNet/fc0/weights:0")
            tf.histogram_summary("fc0/weights", H)
            H = tf.get_default_graph().get_tensor_by_name("PoseNet/fc0/biases:0")
            tf.histogram_summary("fc0/biases", H)
            H = tf.get_default_graph().get_tensor_by_name("PoseNet/fc1/weights:0")
            tf.histogram_summary("fc1/weights", H)
            H = tf.get_default_graph().get_tensor_by_name("PoseNet/fc1/biases:0")
            tf.histogram_summary("fc1/biases", H)

        #print map(lambda x: x.name, tf.get_collection(tf.GraphKeys.VARIABLES, scope='PoseNet'))
        #with tf.variable_scope("PoseNet"):
        #    print tf.get_variable("Inception_V3/Mixed_5c/Branch_1/Conv_1_0c_5x5/weights:0")

        return total_loss, outputs