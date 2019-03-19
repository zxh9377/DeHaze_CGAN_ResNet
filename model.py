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
        layers = []

        encoder_filter_nums = [
            flags.FLAGS.ngf,  # encode_1: [batch_size, 256, 256, img_channels] => [batch_size, 128, 128, ngf]
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
            conv = tf.layers.conv2d(inputs=input_img, filters=encoder_filter_nums[0], kernel_size=4,
                                    strides=(2, 2), padding="same")  # conv
            normal = tf.layers.batch_normalization(inputs=conv, training=self.training)  # batch normalization
            lrelu = tf.nn.leaky_relu(normal)  # lrelu activate
            layers.append(lrelu)

        # encoder_2 ~ encoder_8
        for i in range(len(encoder_filter_nums))[1:]:
            with tf.variable_scope("encoder_{}".format(i + 1)):
                conv = tf.layers.conv2d(inputs=layers[-1], filters=encoder_filter_nums[i], kernel_size=4,
                                        strides=(2, 2), padding="same")  # conv
                normal = tf.layers.batch_normalization(inputs=conv, training=self.training)  # batch normalization
                lrelu = tf.nn.leaky_relu(normal)  # lrelu activate
                layers.append(lrelu)

        num_encode_layers = len(layers)  # the number of encode layers

        encoder_filter_nums = [
            flags.FLAGS.ngf * 8,  # decode_1: [batch_size, 1, 1, ngf*8] => [batch_size, 2, 2, ngf*8]
            flags.FLAGS.ngf * 8,  # decode_2: [batch_size, 2, 2, ngf*8*2] => [batch_size, 4, 4, ngf*8]
            flags.FLAGS.ngf * 8,  # decode_3: [batch_size, 4, 4, ngf*8*2] => [batch_size, 8, 8, ngf*8]
            flags.FLAGS.ngf * 8,  # decode_4: [batch_size, 8, 8, ngf*8*2] => [batch_size, 16, 16, ngf*8]
            flags.FLAGS.ngf * 4,  # decode_5: [batch_size, 16, 16, ngf*8*2] => [batch_size, 32, 32, ngf*4]
            flags.FLAGS.ngf * 2,  # decode_6: [batch_size, 32, 32, ngf*4*2] => [batch_size, 64, 64, ngf*2]
            flags.FLAGS.ngf,  # decode_7: [batch_size, 64, 64, ngf*2*2] => [batch_size, 128, 128, ngf]
            flags.FLAGS.img_channels  # decode_8: [batch_size, 128, 128, ngf*2] => [batch_size, 256, 256, img_channels]
        ]

        # decode_1
        # first decoder layer does not have skip connection
        with tf.variable_scope("decoder_1"):
            deconv = tf.layers.conv2d_transpose(inputs=layers[-1], filters=encoder_filter_nums[0], kernel_size=4,
                                                strides=(2, 2), padding="same")  # deconv
            normal = tf.layers.batch_normalization(inputs=deconv, training=self.training)  # batch normalization
            dropout = tf.nn.dropout(x=normal, keep_prob=self.keep_prob)  # dropout
            lrelu = tf.nn.leaky_relu(dropout)  # lrelu activate
            layers.append(lrelu)

        # decode_2 ~ decode_8
        # these decoder layers have skip connection
        for i in range(len(encoder_filter_nums))[1:]:
            with tf.variable_scope("decoder_{}".format(i + 1)):
                skip_conn = num_encode_layers - i - 1
                conn = tf.concat([layers[-1], layers[skip_conn]], axis=3)
                deconv = tf.layers.conv2d_transpose(inputs=conn, filters=encoder_filter_nums[i], kernel_size=4,
                                                    strides=(2, 2), padding="same")  # deconv
                normal = tf.layers.batch_normalization(inputs=deconv, training=self.training)  # batch normalization
                dropout = tf.nn.dropout(x=normal, keep_prob=self.keep_prob)  # dropout
                lrelu = tf.nn.leaky_relu(dropout)  # lrelu activate
                layers.append(lrelu)

        return layers[-1]

    def create_discriminator(self, input_img, target_img):

        input_img = tf.concat([input_img, target_img], axis=3)

        # conv_1: [batch_size, 256, 256, in_channels*2] => [batch_size, 128, 128, ndf]
        with tf.variable_scope("conv_1"):
            conv1 = tf.layers.conv2d(inputs=input_img, filters=flags.FLAGS.ndf, kernel_size=4, strides=(2, 2),
                                     padding="same")
            relu1 = tf.nn.relu(conv1)

        # conv_2: [batch_size, 128, 128, ndf] => [batch_size, 64, 64, ndf*2]
        with tf.variable_scope("conv_2"):
            conv2 = tf.layers.conv2d(inputs=relu1, filters=flags.FLAGS.ndf * 2, kernel_size=4, strides=(2, 2),
                                     padding="same")
            relu2 = tf.nn.relu(conv2)

        # conv_3: [batch_size, 64, 64, ndf*2] => [batch_size, 32, 32, ndf*4]
        with tf.variable_scope("conv_3"):
            conv3 = tf.layers.conv2d(inputs=relu2, filters=flags.FLAGS.ndf * 4, kernel_size=4, strides=(2, 2),
                                     padding="same")
            relu3 = tf.nn.relu(conv3)

        # conv_4: [batch_size, 32, 32, ndf*4] => [batch_size, 32, 32, ndf*8]
        with tf.variable_scope("conv_4"):
            conv4 = tf.layers.conv2d(inputs=relu3, filters=flags.FLAGS.ndf * 8, kernel_size=4, strides=(1, 1),
                                     padding="same")
            relu4 = tf.nn.relu(conv4)

        # conv_5: [batch_size, 32, 32, ndf*8] => [batch_size, 32, 32, 1]
        with tf.variable_scope("conv_5"):
            conv5 = tf.layers.conv2d(inputs=relu4, filters=1, kernel_size=4, strides=(1, 1), padding="same")
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
