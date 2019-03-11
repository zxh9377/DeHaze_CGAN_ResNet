import tensorflow as tf

tf.flags.DEFINE_string("input_dir", "big-paper-MATLAB/input", "path to folder containing images")
tf.flags.DEFINE_string("output_dir", "big-paper-MATLAB/output", "where to put output files")
tf.flags.DEFINE_string("mode", "test", "model to train or test")

tf.flags.DEFINE_integer("seed", 0, "?")
tf.flags.DEFINE_string("checkpoint", "checkpoint",
                       "directory with checkpoint to resume training from or use for testing")

tf.flags.DEFINE_integer("summary_freq", 100, "update summaries every summary_freq steps")
tf.flags.DEFINE_integer("progress_freq", 50, "number of training epochs")
tf.flags.DEFINE_integer("trace_freq", 0, "trace execution every trace_freq steps")
tf.flags.DEFINE_integer("save_freq", 5000, "save model every save_freq steps, 0 to disable")

tf.flags.DEFINE_float("aspect_ratio", 1.0, "aspect ratio of output images (width/height)")
tf.flags.DEFINE_boolean("lab_colorization", False, "split input image into brightness (A) and color (B)")

tf.flags.DEFINE_string("which_direction", "BtoA", "AtoB or BtoA")
tf.flags.DEFINE_integer("ngf", 64, "number of generator filters in first conv layer")
tf.flags.DEFINE_integer("ndf", 64, "number of discriminator filters in first conv layer")
tf.flags.DEFINE_integer("scale_size", 256, "scale images to this size before cropping to 256x256")
tf.flags.DEFINE_boolean("flip", True, "flip images horizontally")

tf.flags.DEFINE_integer("max_steps", 20000, "number of training steps (0 to disable)")
tf.flags.DEFINE_integer("max_epochs", 100, "number of training epochs")
tf.flags.DEFINE_integer("batch_size", 5, "number of images in batch")
tf.flags.DEFINE_float("lr", 0.0002, "initial learning rate for adam")
tf.flags.DEFINE_float("beta1", 0.5, "initial learning rate for adam")
tf.flags.DEFINE_float("l1_weight", 100.0, "weight on L1 term for generator gradient")
tf.flags.DEFINE_float("gan_weight", 1.0, "weight on GAN term for generator gradient")

tf.flags.DEFINE_string("output_filetype", "png", "png or jpeg")

FLAGS = tf.flags.FLAGS
