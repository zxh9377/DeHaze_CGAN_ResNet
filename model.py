import tensorflow as tf
import flags
import preceptual_loss as p_loss
import data_generator


class Model:

    def __init__(self):
        self.generator_output = None
        self.discriminator_output_gene = None
        self.discriminator_output_clear = None
        self.generator_GAN_loss = None
        self.generator_L1_loss = None
        self.generator_L2_loss = None
        self.generator_p_content_loss = None
        self.generator_p_style_loss = None
        self.generator_loss = None
        self.discriminator_loss = None
        self.generator_train = None
        self.discriminator_train = None
        with tf.variable_scope("input"):
            self.input_clear = tf.placeholder(dtype=tf.float32,
                                              shape=[None, flags.FLAGS.scale_size, flags.FLAGS.scale_size,
                                                     flags.FLAGS.img_channels],
                                              name="input_clear")
            self.input_hazy = tf.placeholder(dtype=tf.float32,
                                             shape=[None, flags.FLAGS.scale_size, flags.FLAGS.scale_size,
                                                    flags.FLAGS.img_channels],
                                             name="input_hazy")
            self.training = tf.placeholder_with_default(input=False, shape=(), name="is_training")
            self.keep_prob = tf.placeholder(dtype=tf.float32, name="keep_prob")
        self.build_model()

    def build_model(self):
        # Generator
        with tf.variable_scope("generator"):
            self.generator_output = self.create_generator(self.input_hazy)

        # Discriminator
        with tf.variable_scope("discriminator"):
            with tf.variable_scope("gene_discriminator"):
                self.discriminator_output_gene = self.create_discriminator(self.input_hazy, self.generator_output)
            with tf.variable_scope("clear_discriminator"):
                self.discriminator_output_clear = self.create_discriminator(self.input_hazy, self.input_clear)

        # loss
        with tf.variable_scope("loss"):
            with tf.variable_scope("generator_loss"):
                gene_losses = self.calc_generator_loss(self.input_hazy,
                                                       self.generator_output,
                                                       self.input_clear,
                                                       self.discriminator_output_gene)
                self.generator_GAN_loss = gene_losses["gene_loss_GAN"]
                self.generator_L1_loss = gene_losses["gene_loss_L1"]
                self.generator_L2_loss = gene_losses["gene_loss_L2"]
                self.generator_p_content_loss = gene_losses["gene_p_content_loss"]
                self.generator_p_style_loss = gene_losses["gene_p_style_loss"]
                self.generator_loss = self.generator_GAN_loss * flags.FLAGS.gene_loss_gan_weight \
                                      + self.generator_L1_loss * flags.FLAGS.gene_loss_l1_weight \
                                      + self.generator_L2_loss * flags.FLAGS.gene_loss_l2_weight \
                                      + self.generator_p_content_loss * flags.FLAGS.gene_p_content_loss_weight \
                                      + self.generator_p_style_loss * flags.FLAGS.gene_p_style_loss_weight
                tf.summary.scalar(name="generator_GAN_loss", tensor=self.generator_GAN_loss)
                tf.summary.scalar(name="generator_L1_loss", tensor=self.generator_L1_loss)
                tf.summary.scalar(name="generator_L2_loss", tensor=self.generator_L2_loss)
                tf.summary.scalar(name="generator_p_content_loss", tensor=self.generator_p_content_loss)
                tf.summary.scalar(name="generator_p_style_loss", tensor=self.generator_p_style_loss)
                tf.summary.scalar(name="generator_loss", tensor=self.generator_loss)
            with tf.variable_scope("discriminator_loss"):
                self.discriminator_loss = self.calc_discriminator_loss(self.discriminator_output_gene,
                                                                       self.discriminator_output_clear)
                tf.summary.scalar(name="discriminator_loss", tensor=self.discriminator_loss)

        # train
        with tf.variable_scope("train"):
            with tf.variable_scope("generator_train"):
                gene_vars = [var for var in tf.trainable_variables() if var.name.startswith("generator")]
                gene_optim = tf.train.AdamOptimizer(flags.FLAGS.gene_learning_rate)
                gene_grad = gene_optim.compute_gradients(self.generator_loss, var_list=gene_vars)
                self.generator_train = gene_optim.apply_gradients(gene_grad)
            with tf.variable_scope("discriminator_train"):
                discrim_vars = [var for var in tf.trainable_variables() if var.name.startswith("discriminator")]
                discrim_optim = tf.train.AdamOptimizer(flags.FLAGS.discrim_learning_rate)
                discrim_grad = discrim_optim.compute_gradients(self.discriminator_loss, var_list=discrim_vars)
                self.discriminator_train = discrim_optim.apply_gradients(discrim_grad)

    def create_generator(self, input_img):
        def _encoder_block(input, kernel_size, stride, out_channels):
            conv = tf.layers.conv2d(inputs=input, filters=out_channels, kernel_size=kernel_size, strides=stride,
                                    padding="same")  # conv
            normal = tf.layers.batch_normalization(inputs=conv, training=self.training)  # batch normalization
            lrelu = tf.nn.leaky_relu(normal)  # lrelu activate
            return lrelu

        def _decoder_block(input, kernel_size, stride, out_channels, keep_prob):
            deconv = tf.layers.conv2d_transpose(inputs=input, filters=out_channels, kernel_size=kernel_size,
                                                strides=stride, padding="same")  # deconv
            normal = tf.layers.batch_normalization(inputs=deconv, training=self.training)  # batch normalization
            dropout = tf.nn.dropout(x=normal, keep_prob=keep_prob)  # dropout
            relu = tf.nn.relu(dropout)  # relu activate
            return relu

        layers = []

        encoder_filter_num = [
            flags.FLAGS.ngf * 1,  # encoder_1: [batch_size, 256, 256, img_channels] => [batch_size, 256, 256, ngf*1]
            flags.FLAGS.ngf * 1,  # encoder_2: [batch_size, 256, 256, ngf*1] => [batch_size, 256, 256, ngf*1]
            flags.FLAGS.ngf * 1,  # encoder_3: [batch_size, 256, 256, ngf*1] => [batch_size, 128, 128, ngf*1]
            flags.FLAGS.ngf * 2,  # encoder_4: [batch_size, 128, 128, ngf*1] => [batch_size, 64, 64, ngf*2]
            flags.FLAGS.ngf * 4,  # encoder_5: [batch_size, 64, 64, ngf*2] => [batch_size, 32, 32, ngf*4]
            flags.FLAGS.ngf * 8,  # encoder_6: [batch_size, 32, 32, ngf*4] => [batch_size, 16, 16, ngf*8]
            flags.FLAGS.ngf * 16,  # encoder_7: [batch_size, 16, 16, ngf*8] => [batch_size, 8, 8, ngf*16]
            flags.FLAGS.ngf * 16,  # encoder_8: [batch_size, 8, 8, ngf*16] => [batch_size, 4, 4, ngf*16]
            flags.FLAGS.ngf * 16,  # encoder_9: [batch_size, 4, 4, ngf*16] => [batch_size, 2, 2, ngf*16]
            flags.FLAGS.ngf * 16  # encoder_10: [batch_size, 2, 2, ngf*16] => [batch_size, 1, 1, ngf*16]
        ]

        with tf.variable_scope("encoder_1"):
            layers.append(_encoder_block(input_img, kernel_size=5, stride=(1, 1), out_channels=encoder_filter_num[0]))
        with tf.variable_scope("encoder_2"):
            layers.append(_encoder_block(layers[-1], kernel_size=3, stride=(1, 1), out_channels=encoder_filter_num[1]))
        with tf.variable_scope("encoder_3"):
            layers.append(_encoder_block(layers[-1], kernel_size=4, stride=(2, 2), out_channels=encoder_filter_num[2]))
        with tf.variable_scope("encoder_4"):
            layers.append(_encoder_block(layers[-1], kernel_size=4, stride=(2, 2), out_channels=encoder_filter_num[3]))
        with tf.variable_scope("encoder_5"):
            layers.append(_encoder_block(layers[-1], kernel_size=4, stride=(2, 2), out_channels=encoder_filter_num[4]))
        with tf.variable_scope("encoder_6"):
            layers.append(_encoder_block(layers[-1], kernel_size=4, stride=(2, 2), out_channels=encoder_filter_num[5]))
        with tf.variable_scope("encoder_7"):
            layers.append(_encoder_block(layers[-1], kernel_size=4, stride=(2, 2), out_channels=encoder_filter_num[6]))
        with tf.variable_scope("encoder_8"):
            layers.append(_encoder_block(layers[-1], kernel_size=4, stride=(2, 2), out_channels=encoder_filter_num[7]))
        with tf.variable_scope("encoder_9"):
            layers.append(_encoder_block(layers[-1], kernel_size=4, stride=(2, 2), out_channels=encoder_filter_num[8]))
        with tf.variable_scope("encoder_10"):
            layers.append(_encoder_block(layers[-1], kernel_size=4, stride=(2, 2), out_channels=encoder_filter_num[9]))

        num_encode_layers = len(layers)  # the number of encode layers

        decoder_filter_num = [
            flags.FLAGS.ngf * 16,  # decoder_1: [batch_size, 1, 1, ngf*16] => [batch_size, 2, 2, ngf*16]
            flags.FLAGS.ngf * 16,  # decoder_2: [batch_size, 2, 2, ngf*16*2] => [batch_size, 4, 4, ngf*16]
            flags.FLAGS.ngf * 16,  # decoder_3: [batch_size, 4, 4, ngf*16*2] => [batch_size, 8, 8, ngf*16]
            flags.FLAGS.ngf * 8,  # decoder_4: [batch_size, 8, 8, ngf*16*2] => [batch_size, 16, 16, ngf*8]
            flags.FLAGS.ngf * 4,  # decoder_5: [batch_size, 16, 16, ngf*8*2] => [batch_size, 32, 32, ngf*4]
            flags.FLAGS.ngf * 2,  # decoder_6: [batch_size, 32, 32, ngf*4*2] => [batch_size, 64, 64, ngf*2]
            flags.FLAGS.ngf * 1,  # decoder_7: [batch_size, 64, 64, ngf*2*2] => [batch_size, 128, 128, ngf*1]
            flags.FLAGS.ngf * 1,  # decoder_8: [batch_size, 128, 128, ngf*1*2] => [batch_size, 256, 256, ngf*1]
            flags.FLAGS.img_channels  # decoder_9: [batch_size, 256, 256, ngf*1*2]=>[batch_size, 256, 256, img_channels]
        ]

        # first decoder layer does not have skip connection
        with tf.variable_scope("decoder_1"):
            layers.append(_decoder_block(layers[-1], kernel_size=4, stride=(2, 2),
                                         out_channels=decoder_filter_num[0], keep_prob=self.keep_prob))
        with tf.variable_scope("decoder_2"):
            decoder_input = tf.concat([layers[-1], layers[num_encode_layers - 2]], axis=3)
            layers.append(_decoder_block(decoder_input, kernel_size=4, stride=(2, 2),
                                         out_channels=decoder_filter_num[1], keep_prob=self.keep_prob))
        with tf.variable_scope("decoder_3"):
            decoder_input = tf.concat([layers[-1], layers[num_encode_layers - 3]], axis=3)
            layers.append(_decoder_block(decoder_input, kernel_size=4, stride=(2, 2),
                                         out_channels=decoder_filter_num[2], keep_prob=self.keep_prob))
        with tf.variable_scope("decoder_4"):
            decoder_input = tf.concat([layers[-1], layers[num_encode_layers - 4]], axis=3)
            layers.append(_decoder_block(decoder_input, kernel_size=4, stride=(2, 2),
                                         out_channels=decoder_filter_num[3], keep_prob=self.keep_prob))
        with tf.variable_scope("decoder_5"):
            decoder_input = tf.concat([layers[-1], layers[num_encode_layers - 5]], axis=3)
            layers.append(_decoder_block(decoder_input, kernel_size=4, stride=(2, 2),
                                         out_channels=decoder_filter_num[4], keep_prob=self.keep_prob))
        with tf.variable_scope("decoder_6"):
            decoder_input = tf.concat([layers[-1], layers[num_encode_layers - 6]], axis=3)
            layers.append(_decoder_block(decoder_input, kernel_size=4, stride=(2, 2),
                                         out_channels=decoder_filter_num[5], keep_prob=self.keep_prob))
        with tf.variable_scope("decoder_7"):
            decoder_input = tf.concat([layers[-1], layers[num_encode_layers - 7]], axis=3)
            layers.append(_decoder_block(decoder_input, kernel_size=4, stride=(2, 2),
                                         out_channels=decoder_filter_num[6], keep_prob=self.keep_prob))
        with tf.variable_scope("decoder_8"):
            decoder_input = tf.concat([layers[-1], layers[num_encode_layers - 8]], axis=3)
            layers.append(_decoder_block(decoder_input, kernel_size=4, stride=(2, 2),
                                         out_channels=decoder_filter_num[7], keep_prob=self.keep_prob))
        # the last decoder is a conv not deconv
        with tf.variable_scope("decoder_9"):
            decoder_input = tf.concat([layers[-1], layers[num_encode_layers - 9]], axis=3)
            conv = tf.layers.conv2d(inputs=decoder_input, filters=decoder_filter_num[8], kernel_size=3, strides=(1, 1),
                                    padding="same")
            normal = tf.layers.batch_normalization(inputs=conv, training=self.training)
            tanh = tf.nn.tanh(normal)
            layers.append(tanh)

        return layers[-1]

    def create_discriminator(self, input_img, target_img):
        input_img = tf.concat([input_img, target_img], axis=3)

        # conv_1: [batch_size, 256, 256, in_channels*2] => [batch_size, 256, 256, ndf]
        with tf.variable_scope("conv_1"):
            conv1 = tf.layers.conv2d(inputs=input_img, filters=flags.FLAGS.ndf, kernel_size=3, strides=(1, 1),
                                     padding="same")
            relu1 = tf.nn.relu(conv1)

        # conv_2: [batch_size, 256, 256, ndf] => [batch_size, 256, 256, ndf*2]
        with tf.variable_scope("conv_2"):
            conv2 = tf.layers.conv2d(inputs=relu1, filters=flags.FLAGS.ndf * 2, kernel_size=3, strides=(1, 1),
                                     padding="same")
            relu2 = tf.nn.relu(conv2)

        # conv_3: [batch_size, 256, 256, ndf*2] => [batch_size, 256, 256, ndf*4]
        with tf.variable_scope("conv_3"):
            conv3 = tf.layers.conv2d(inputs=relu2, filters=flags.FLAGS.ndf * 4, kernel_size=3, strides=(1, 1),
                                     padding="same")
            relu3 = tf.nn.relu(conv3)

        # conv_4: [batch_size, 256, 256, ndf*4] => [batch_size, 256, 256, ndf*8]
        with tf.variable_scope("conv_4"):
            conv4 = tf.layers.conv2d(inputs=relu3, filters=flags.FLAGS.ndf * 8, kernel_size=3, strides=(1, 1),
                                     padding="same")
            relu4 = tf.nn.relu(conv4)

        # conv_5: [batch_size, 256, 256, ndf*8] => [batch_size, 256, 256, 1]
        with tf.variable_scope("conv_5"):
            conv5 = tf.layers.conv2d(inputs=relu4, filters=1, kernel_size=3, strides=(1, 1), padding="same")
            sigmod5 = tf.nn.sigmoid(conv5)

        return sigmod5

    def calc_generator_loss(self, hazy_img, gene_img, clear_img, discrim_out_gene):
        gene_loss_GAN = tf.reduce_mean(-tf.log(discrim_out_gene + flags.FLAGS.EPS))
        gene_loss_L1 = tf.reduce_mean(tf.abs(gene_img - clear_img))
        gene_loss_L2 = tf.reduce_mean(tf.nn.l2_loss(gene_img - clear_img))
        gene_p_content_loss, gene_p_style_loss = p_loss.calc_preceptual_loss(hazy_img, gene_img, clear_img)
        return {
            "gene_loss_GAN": gene_loss_GAN,
            "gene_loss_L1": gene_loss_L1,
            "gene_loss_L2": gene_loss_L2,
            "gene_p_content_loss": gene_p_content_loss,
            "gene_p_style_loss": gene_p_style_loss
        }

    def calc_discriminator_loss(self, discrim_out_gene, discrim_out_clear):
        discrim_loss = tf.reduce_mean(
            -(tf.log(discrim_out_clear + flags.FLAGS.EPS) + tf.log(1 - discrim_out_gene + flags.FLAGS.EPS)))
        return discrim_loss

# dataset = data_generator.Data_Generator()
# model = Model()
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     for i in range(20):
#         input = sess.run(dataset.input)
#         feed_dict = {
#             model.input_clear: input[0],
#             model.input_hazy: input[1],
#             model.training: True,
#             model.keep_prob: flags.FLAGS.keep_prob
#         }
#         sess.run(model.discriminator_train, feed_dict=feed_dict)
#         sess.run(model.generator_train, feed_dict=feed_dict)
#         discrim_loss = sess.run(model.discriminator_loss, feed_dict=feed_dict)
#         gene_loss = sess.run(model.generator_loss, feed_dict=feed_dict)
#         print("step {0} : discrimator_loss:{1} genenrator_loss:{2}".format(i, discrim_loss, gene_loss))
