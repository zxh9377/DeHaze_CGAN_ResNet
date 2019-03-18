import tensorflow as tf
import cv2
import numpy
import preceptual_loss as p_loss
import vgg.vgg16 as vgg_16

img = []
img.append(cv2.resize(
    cv2.imread("/home/csbhr/workspace/python/python_data/DeHaze_CGAN_ResNet/train/clear/0001.png", cv2.IMREAD_COLOR),
    (256, 256)))
img.append(cv2.resize(
    cv2.imread("/home/csbhr/workspace/python/python_data/DeHaze_CGAN_ResNet/train/clear/0002.png", cv2.IMREAD_COLOR),
    (256, 256)))
img.append(cv2.resize(
    cv2.imread("/home/csbhr/workspace/python/python_data/DeHaze_CGAN_ResNet/train/clear/0003.png", cv2.IMREAD_COLOR),
    (256, 256)))
img.append(cv2.resize(
    cv2.imread("/home/csbhr/workspace/python/python_data/DeHaze_CGAN_ResNet/train/clear/0004.png", cv2.IMREAD_COLOR),
    (256, 256)))

img_gen = []
img_gen.append(cv2.resize(
    cv2.imread("/home/csbhr/workspace/python/python_data/DeHaze_CGAN_ResNet/train/clear/0006.png", cv2.IMREAD_COLOR),
    (256, 256)))
img_gen.append(cv2.resize(
    cv2.imread("/home/csbhr/workspace/python/python_data/DeHaze_CGAN_ResNet/train/clear/0007.png", cv2.IMREAD_COLOR),
    (256, 256)))
img_gen.append(cv2.resize(
    cv2.imread("/home/csbhr/workspace/python/python_data/DeHaze_CGAN_ResNet/train/clear/0009.png", cv2.IMREAD_COLOR),
    (256, 256)))
img_gen.append(cv2.resize(
    cv2.imread("/home/csbhr/workspace/python/python_data/DeHaze_CGAN_ResNet/train/clear/0010.png", cv2.IMREAD_COLOR),
    (256, 256)))

img_target = []
img_target.append(cv2.resize(
    cv2.imread("/home/csbhr/workspace/python/python_data/DeHaze_CGAN_ResNet/train/clear/0011.png", cv2.IMREAD_COLOR),
    (256, 256)))
img_target.append(cv2.resize(
    cv2.imread("/home/csbhr/workspace/python/python_data/DeHaze_CGAN_ResNet/train/clear/0014.png", cv2.IMREAD_COLOR),
    (256, 256)))
img_target.append(cv2.resize(
    cv2.imread("/home/csbhr/workspace/python/python_data/DeHaze_CGAN_ResNet/train/clear/0016.png", cv2.IMREAD_COLOR),
    (256, 256)))
img_target.append(cv2.resize(
    cv2.imread("/home/csbhr/workspace/python/python_data/DeHaze_CGAN_ResNet/train/clear/0017.png", cv2.IMREAD_COLOR),
    (256, 256)))

img = numpy.array(img)
img_gen = numpy.array(img_gen)
img_target = numpy.array(img_target)

img = tf.constant(img, dtype=tf.uint8)
img_gen = tf.constant(img_gen, dtype=tf.uint8)
img_target = tf.constant(img_target, dtype=tf.uint8)

img = tf.cast(img, dtype=tf.float32)
img_gen = tf.cast(img_gen, dtype=tf.float32)
img_target = tf.cast(img_target, dtype=tf.float32)

vgg_16_model = vgg_16.Vgg16()
vgg

closs, sloss = p_loss.calc_preceptual_loss(img, img_gen, img_target)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    result = sess.run([closs, sloss])
    print()
