import tensorflow as tf
import flags


class Model:

    def __init__(self):
        return

    def conv(self, input, out_channels, filter_size, stride=2):
        with tf.variable_scope("conv"):
            # input => [batch_size, in_height, in_width, in_channels]
            in_channels = input.get_shape()[3]
            # filter => [filter_height, filter_width, in_channels, out_channels]
            filter = tf.get_variable(name="filter",
                                     shape=[filter_size, filter_size, in_channels, out_channels],
                                     dtype=tf.float32,
                                     initializer=tf.random_normal_initializer(mean=0., stddev=0.02))
            conv = tf.nn.conv2d(input=input,
                                filter=filter,
                                strides=[1, stride, stride, 1],
                                padding="SAME")
            return conv

    def deconv(self, input, out_channels, filter_size, stride=2):
        with tf.variable_scope("deconv"):
            # input => [batch_size, in_height, in_width, in_channels]
            batch_size, in_height, in_width, in_channels = input.get_shape()
            # filter => [filter_height, filter_width, out_channels, in_channels]
            filter = tf.get_variable(name="filter",
                                     shape=[filter_size, filter_size, out_channels, in_channels],
                                     dtype=tf.float32,
                                     initializer=tf.random_normal_initializer(mean=0., stddev=0.02))
            deconv = tf.nn.conv2d_transpose(value=input,
                                            filter=filter,
                                            output_shape=[batch_size, in_height * 2, in_width * 2, out_channels],
                                            strides=[1, stride, stride, 1],
                                            padding="SAME")
            return deconv

    def lrelu(self, input, alpha=0.2):
        # if input >= 0 , lrelu(input) = input
        # if input < 0, lrelu(input) = alpha*input
        with tf.name_scope("lrelu"):
            return tf.nn.leaky_relu(features=input, alpha=alpha)

    def batch_normal(self):
