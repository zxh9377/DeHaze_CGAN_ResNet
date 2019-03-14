import tensorflow as tf
import os
import glob
import cv2
import flags


class Data_Generator:

    def __init__(self):
        self.dataset = None
        self.data_iterator = None
        self.input = None
        if flags.FLAGS.mode == "train" or "test":  # train and test mode have the same dataset structure
            self.load_train_dataset()
        elif flags.FLAGS.mode == "use":
            self.load_use_dataset()
        else:
            raise Exception("Message: no support mode!")

    # load train dataset
    def load_train_dataset(self):

        # load clear and hazy image paths for couple
        def _load_couple_img_path(clear_dir, hazy_dir):
            self.check_path(clear_dir, message="dir({}) is not exist!".format(clear_dir))
            self.check_path(hazy_dir, message="dir({}) is not exist!".format(hazy_dir))

            # get all clear images' filename from clear_dir
            clear_img_fname_list = os.listdir(clear_dir)

            couple_clear_img = []
            couple_hazy_img = []
            # match clear and hazy images for couple
            for clear_fname in clear_img_fname_list:
                base_name = clear_fname.split(".")[0]  # get basename
                hazy_imgs = glob.glob(os.path.join(hazy_dir, base_name + "*"))  # match the hazy image
                if len(hazy_imgs) == 0:  # there is no matched hazy image
                    continue
                clear_path = os.path.join(clear_dir, clear_fname)  # clear image's full path
                hazy_path = hazy_imgs[0]  # hazy image's full path
                couple_clear_img.append(clear_path)
                couple_hazy_img.append(hazy_path)

            # there is no couple clear and hazy image
            if len(couple_hazy_img) == 0:
                raise Exception("Message: there is no couple clear and hazy image!")

            return couple_clear_img, couple_hazy_img

        # read image
        def _read_image(clear_img_path, hazy_img_path):
            # for tf.py_func() to call opencv module
            def _for_py_func(clear_img_path, hazy_img_path):
                clear_img = cv2.imread(clear_img_path.decode(), cv2.IMREAD_COLOR)  # read image
                hazy_img = cv2.imread(hazy_img_path.decode(), cv2.IMREAD_COLOR)
                return clear_img, hazy_img

            # use tf.py_func() to call opencv module
            return tf.py_func(func=_for_py_func, inp=[clear_img_path, hazy_img_path],
                              Tout=[tf.uint8, tf.uint8])

        # preprocess image
        def _preprocess_image(clear_img, hazy_img):
            clear_img = self.preprocess_image(clear_img)
            hazy_img = self.preprocess_image(hazy_img)
            return clear_img, hazy_img

        couple_clear_img, couple_hazy_img = _load_couple_img_path(clear_dir=flags.FLAGS.train_clear_dir,
                                                                  hazy_dir=flags.FLAGS.train_hazy_dir)

        self.dataset = tf.data.Dataset.from_tensor_slices((couple_clear_img, couple_hazy_img))
        self.dataset = self.dataset.map(map_func=_read_image)  # read image
        self.dataset = self.dataset.map(map_func=_preprocess_image)  # preprocess image
        if flags.FLAGS.shuffle:
            self.dataset = self.dataset.shuffle(buffer_size=200)  # shuffle
        self.dataset = self.dataset.batch(batch_size=flags.FLAGS.batch_size)  # bacth
        self.dataset = self.dataset.repeat(count=flags.FLAGS.epoch_num)  # epoch

        self.data_iterator = self.dataset.make_one_shot_iterator()
        self.input = self.data_iterator.get_next()

    def load_use_dataset(self):

        # read image
        def _read_image(img_path):
            # for tf.py_func() to call opencv module
            def _for_py_func(img_path):
                hazy_img = cv2.imread(img_path.decode(), cv2.IMREAD_COLOR)  # read image
                return hazy_img

            # use tf.py_func() to call opencv module
            return tf.py_func(func=_for_py_func, inp=[hazy_img_path], Tout=[tf.uint8])

        hazy_img_fname_list = os.listdir(flags.FLAGS.use_hazy_dir)  # get hazy images' filename
        hazy_img_path = [os.path.join(flags.FLAGS.use_hazy_dir, fname) for fname in hazy_img_fname_list]  # full path

        self.dataset = tf.data.Dataset.from_tensor_slices(hazy_img_path)
        self.dataset = self.dataset.map(map_func=_read_image)  # read image
        self.dataset = self.dataset.map(map_func=self.preprocess_image)  # preprocess image
        self.dataset = self.dataset.batch(batch_size=flags.FLAGS.batch_size)  # bacth

        self.data_iterator = self.dataset.make_one_shot_iterator()
        self.input = self.data_iterator.get_next()

    # preprocess single image
    def preprocess_image(self, img):
        img.set_shape([None, None, None])  # tensorflow can not infer the image's shape, so must set shape
        img = tf.image.resize_images(img, [flags.FLAGS.scale_size, flags.FLAGS.scale_size])  # resize
        img = tf.cast(img, tf.float32) / 255.  # convert [0,255]=>[0,1]
        img = img * 2. - 1.  # convert [0,1]=>[-1,1]
        return img

    # check if the path exist
    def check_path(self, path, message=None):
        if path is None or not os.path.exists(path):
            raise Exception("Message: {}".format(message))


a = Data_Generator()
with tf.Session() as sess:
    result = sess.run(a.input)
    print(result)
