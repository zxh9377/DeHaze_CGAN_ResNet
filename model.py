import tensorflow as tf
from tensorflow.python.training import moving_averages
import flags


class Model:

    def __init__(self):
        with tf.variable_scope("input"):
            self.input_clear = tf.placeholder(dtype=tf.float32,
                                              shape=[None, flags.FLAGS.scale_size, flags.FLAGS.scale_size, 3],
                                              name="input_clear")
            self.input_hazy = tf.placeholder(dtype=tf.float32,
                                             shape=[None, flags.FLAGS.scale_size, flags.FLAGS.scale_size, 3],
                                             name="input_hazy")
            self.training = tf.placeholder_with_default(input=False, shape=(), name="is_training")

    def create_generator(self):
        with tf.variable_scope("generator"):
            layers = []

            encoder_filter_nums = [
                flags.FLAGS.ngf,  # encode_1: [batch_size, 256, 256, 3] => [batch_size, 128, 128, ngf]
                flags.FLAGS.ngf * 2,  # encode_2: [batch_size, 128, 128, ngf] => [batch_size, 64, 64, ngf*2]
                flags.FLAGS.ngf * 4,  # encode_3: [batch_size, 64, 64, ngf*2] => [batch_size, 32, 32, ngf*4]
                flags.FLAGS.ngf * 8,  # encode_4: [batch_size, 32, 32, ngf*4] => [batch_size, 16, 16, ngf*8]
                flags.FLAGS.ngf * 8,  # encode_5: [batch_size, 16, 16, ngf*8] => [batch_size, 8, 8, ngf*8]
                flags.FLAGS.ngf * 8,  # encode_6: [batch_size, 8, 8, ngf*8] => [batch_size, 4, 4, ngf*8]
                flags.FLAGS.ngf * 8,  # encode_7: [batch_size, 4, 4, ngf*8] => [batch_size, 2, 2, ngf*8]
                flags.FLAGS.ngf * 8  # encode_8: [batch_size, 2, 2, ngf*8] => [batch_size, 1, 1, ngf*8]
            ]

            # encoder_1
            with tf.variable_scope("encoder_1"):
                conv = tf.layers.conv2d(inputs=self.input_hazy, filters=encoder_filter_nums[0], kernel_size=4,
                                        strides=(2, 2), padding="same")
                normal = tf.layers.batch_normalization(inputs=conv, training=self.training)
                lrelu = tf.nn.leaky_relu(normal)
                layers.append(lrelu)

            # encoder_2 ~ encoder_8
            for i in range(len(encoder_filter_nums))[1:]:
                with tf.variable_scope("encoder_{}".format(i + 1)):
                    conv = tf.layers.conv2d(inputs=layers[-1], filters=encoder_filter_nums[i], kernel_size=4,
                                            strides=(2, 2), padding="same")
                    normal = tf.layers.batch_normalization(inputs=conv, training=self.training)
                    if not i == len(encoder_filter_nums) - 1:
                        lrelu = tf.nn.leaky_relu(normal)
                        layers.append(lrelu)
                    else:
                        layers.append(normal)

            encoder_filter_nums = [
                (flags.FLAGS.ngf * 8, 0.5),  # decode_1: [batch_size, 1, 1, ngf*8] => [batch_size, 2, 2, ngf*8]
                (flags.FLAGS.ngf * 8, 0.5),  # decode_2: [batch_size, 2, 2, ngf*8*2] => [batch_size, 4, 4, ngf*8]
                (flags.FLAGS.ngf * 8, 0.5),  # decode_3: [batch_size, 4, 4, ngf*8*2] => [batch_size, 8, 8, ngf*8]
                (flags.FLAGS.ngf * 8, 0.),  # decode_4: [batch_size, 8, 8, ngf*8*2] => [batch_size, 16, 16, ngf*8]
                (flags.FLAGS.ngf * 4, 0.),  # decode_5: [batch_size, 16, 16, ngf*8*2] => [batch_size, 32, 32, ngf*4]
                (flags.FLAGS.ngf * 2, 0.),  # decode_6: [batch_size, 32, 32, ngf*4*2] => [batch_size, 64, 64, ngf*2]
                (flags.FLAGS.ngf, 0.),  # decode_7: [batch_size, 64, 64, ngf*2*2] => [batch_size, 128, 128, ngf]
                (1, 0.)  # decode_8: [batch_size, 128, 128, ngf*2] => [batch_size, 256, 256, 1]
            ]

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

    def batch_normal(self, input):
        with tf.variable_scope("batch_normal"):
            channels = input.get_shape()[3]  # the number of channels

            offset = tf.get_variable(name="offset", shape=[channels], dtype=tf.float32,
                                     initializer=tf.zeros_initializer())
            scale = tf.get_variable(name="scale", shape=[channels], dtype=tf.float32, initializer=tf.ones_initializer())

            mean, variance = tf.nn.moments(x=input, axes=[0, 1, 2])
            variance_epsilon = 1e-3

            normalized = tf.nn.batch_normalization(x=input, mean=mean, variance=variance, offset=offset, scale=scale,
                                                   variance_epsilon=variance_epsilon)
            return normalized
