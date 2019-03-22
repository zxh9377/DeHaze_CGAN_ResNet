import tensorflow as tf
import os
import glob
import cv2
import flags


class Data_Generator:

    def __init__(self, mode):
        self.dataset = None
        self.data_iterator = None
        self.input = None
        if mode == "train":
            self.load_train_dataset()
        elif mode == "val":
            self.load_val_dataset()
        elif mode == "use":
            self.load_use_dataset()
        else:
            raise Exception("Message: no support mode!")

    # load train dataset
    def load_train_dataset(self):

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
            clear_img = Data_Generator.preprocess_image(clear_img)
            hazy_img = Data_Generator.preprocess_image(hazy_img)
            return clear_img, hazy_img

        couple_clear_img, couple_hazy_img = Data_Generator.load_couple_img_path(clear_dir=flags.FLAGS.train_clear_dir,
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

    # load val dataset
    def load_val_dataset(self):

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
            clear_img = Data_Generator.preprocess_image(clear_img)
            hazy_img = Data_Generator.preprocess_image(hazy_img)
            return clear_img, hazy_img

        couple_clear_img, couple_hazy_img = Data_Generator.load_couple_img_path(clear_dir=flags.FLAGS.val_clear_dir,
                                                                                hazy_dir=flags.FLAGS.val_hazy_dir)

        self.dataset = tf.data.Dataset.from_tensor_slices((couple_clear_img, couple_hazy_img))
        self.dataset = self.dataset.map(map_func=_read_image)  # read image
        self.dataset = self.dataset.map(map_func=_preprocess_image)  # preprocess image
        if flags.FLAGS.shuffle:
            self.dataset = self.dataset.shuffle(buffer_size=200)  # shuffle
        self.dataset = self.dataset.batch(batch_size=flags.FLAGS.val_batch_size)  # bacth
        self.dataset = self.dataset.repeat()  # epoch

        self.data_iterator = self.dataset.make_one_shot_iterator()
        self.input = self.data_iterator.get_next()

    # load test dataset
    def load_use_dataset(self):

        # read image
        def _read_image(img_path):
            # for tf.py_func() to call opencv module
            def _for_py_func(img_path):
                hazy_img = cv2.imread(img_path.decode(), cv2.IMREAD_COLOR)  # read image
                return hazy_img

            # use tf.py_func() to call opencv module
            return tf.py_func(func=_for_py_func, inp=[img_path], Tout=[tf.uint8])

        hazy_img_fname_list = os.listdir(flags.FLAGS.use_hazy_dir)  # get hazy images' filename
        hazy_img_path = [os.path.join(flags.FLAGS.use_hazy_dir, fname) for fname in hazy_img_fname_list]  # full path

        self.dataset = tf.data.Dataset.from_tensor_slices(hazy_img_path)
        self.dataset = self.dataset.map(map_func=_read_image)  # read image
        self.dataset = self.dataset.map(map_func=Data_Generator.preprocess_image)  # preprocess image
        self.dataset = self.dataset.batch(batch_size=flags.FLAGS.use_batch_size)  # bacth

        self.data_iterator = self.dataset.make_one_shot_iterator()
        self.input = self.data_iterator.get_next()

    # preprocess single image
    @staticmethod
    def preprocess_image(img):
        img.set_shape([None, None, None])  # tensorflow can not infer the image's shape, so must set shape
        img = tf.image.resize_images(img, [flags.FLAGS.scale_size, flags.FLAGS.scale_size])  # resize
        img = Data_Generator.scale_img(img)
        return img

    # scale image [0,255] => [-1,1]
    @staticmethod
    def scale_img(img):
        img = tf.cast(img, tf.float32) / 255.  # convert [0,255]=>[0,1]
        img = img * 2. - 1.  # convert [0,1]=>[-1,1]
        return img

    # scale image [-1,1] => [0,255]
    @staticmethod
    def de_scale_img(img):
        img = (img + 1) / 2.  # convert [-1,1]=>[0,1]
        img = img * 255.  # convert [0,1]=>[0,255]
        return img

    # check if the path exist
    @staticmethod
    def check_path(path, message=None):
        if path is None or not os.path.exists(path):
            raise Exception("Message: {}".format(message))

    # load clear and hazy image paths for couple
    @staticmethod
    def load_couple_img_path(clear_dir, hazy_dir):
        Data_Generator.check_path(clear_dir, message="dir({}) is not exist!".format(clear_dir))
        Data_Generator.check_path(hazy_dir, message="dir({}) is not exist!".format(hazy_dir))

        clear_img_fname_list = os.listdir(clear_dir)
        hazy_img_fname_list = os.listdir(clear_dir)

        # full path
        couple_clear_img = [os.path.join(clear_dir, cf) for cf in clear_img_fname_list]
        couple_hazy_img = [os.path.join(hazy_dir, hf) for hf in hazy_img_fname_list]

        # there is no couple clear and hazy image
        if len(couple_hazy_img) == 0:
            raise Exception("Message: there is no couple clear and hazy image!")

        return couple_clear_img, couple_hazy_img
