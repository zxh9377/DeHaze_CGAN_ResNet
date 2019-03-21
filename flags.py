import tensorflow as tf

tf.flags.DEFINE_string("mode", "use", "model to train or test or use")

tf.flags.DEFINE_string("train_clear_dir", "/home/csbhr/workspace/python/python_data/DeHaze_CGAN_ResNet/train/clear",
                       "path to folder containing clear images when training this model")
tf.flags.DEFINE_string("train_hazy_dir", "/home/csbhr/workspace/python/python_data/DeHaze_CGAN_ResNet/train/hazy",
                       "path to folder containing hazy images when training this model")
tf.flags.DEFINE_string("val_clear_dir", "/home/csbhr/workspace/python/python_data/DeHaze_CGAN_ResNet/val/clear",
                       "path to folder containing clear images when valing this model")
tf.flags.DEFINE_string("val_hazy_dir", "/home/csbhr/workspace/python/python_data/DeHaze_CGAN_ResNet/val/hazy",
                       "path to folder containing hazy images when valing this model")
tf.flags.DEFINE_string("use_hazy_dir", "/home/csbhr/workspace/python/python_data/DeHaze_CGAN_ResNet/use/hazy",
                       "path to folder containing hazy images when using this system")
tf.flags.DEFINE_string("use_store_dir", "/home/csbhr/workspace/python/python_data/DeHaze_CGAN_ResNet/use/gene",
                       "path to store clear dehazed from hazy images using this system")
tf.flags.DEFINE_string("vgg_model_dir", "/home/csbhr/workspace/python/python_data/DeHaze_CGAN_ResNet/vgg16.npy",
                       "path to store checkpoint")
tf.flags.DEFINE_string("log_dir", "/home/csbhr/workspace/python/python_data/DeHaze_CGAN_ResNet/log",
                       "path to store log files")
tf.flags.DEFINE_string("checkpoint_dir", "/home/csbhr/workspace/python/python_data/DeHaze_CGAN_ResNet/checkpoint",
                       "path to store checkpoint")

tf.flags.DEFINE_integer("img_channels", 3, "the number of the channels of input images")
tf.flags.DEFINE_integer("scale_size", 256, "scale images to this size in preprocess")
tf.flags.DEFINE_boolean("shuffle", True, "if shuffle train dataset")
tf.flags.DEFINE_integer("epoch_num", 10, "number of training epochs")
tf.flags.DEFINE_integer("batch_size", 2, "number of images in batch")
tf.flags.DEFINE_integer("val_batch_size", 2, "number of images in batch when valing this system")
tf.flags.DEFINE_integer("use_batch_size", 1, "number of images in batch when using this system")
tf.flags.DEFINE_integer("max_steps", 50000, "number of training steps (0 to disable)")
tf.flags.DEFINE_integer("save_freq", 100, "save model every save_freq steps, 0 to disable")
tf.flags.DEFINE_integer("val_freq", 10, "val model every val_freq steps")
tf.flags.DEFINE_integer("summary_freq", 10, "update summaries every summary_freq steps")

tf.flags.DEFINE_integer("ngf", 64, "number of generator filters in first conv layer")
tf.flags.DEFINE_integer("ndf", 48, "number of discriminator filters in first conv layer")
# tf.flags.DEFINE_float("keep_prob", 1., "the keep probability in dorpout when training")

tf.flags.DEFINE_float("EPS", 1e-12, "to avoid the input of log function equals zero")
tf.flags.DEFINE_float("gene_loss_gan_weight", 1., "the weight of gan loss in generator loss")
tf.flags.DEFINE_float("gene_loss_l1_weight", 150., "the weight of l1 loss in generator loss")
tf.flags.DEFINE_float("gene_loss_l1_regular", 1e-5, "the factor of regularization in generator L1 loss")
tf.flags.DEFINE_float("gene_p_loss_weight", 150., "the weight of preceptual loss in generator loss")

tf.flags.DEFINE_float("gene_learning_rate", 0.0002, "initial learning rate for adam in generator training")
tf.flags.DEFINE_float("discrim_learning_rate", 0.0002, "initial learning rate for adam int discriminator training")

tf.flags.DEFINE_string("output_filetype", "png", "png or jpeg")

FLAGS = tf.flags.FLAGS
