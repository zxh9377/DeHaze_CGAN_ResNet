import tensorflow as tf
import vgg.vgg16 as vgg_16
import flags
import data_generator


def calc_preceptual_loss(hazy_img, gene_img, clear_img):
    '''calculate preceptual loss

        reference: Perceptual Losses for Real-Time Style Transfer and Super-Resolution

        args:
            hazy_img: [batch_size, height, width, channels]
            gene_img: [batch_size, height, width, channels]
            clear_img: [batch_size, height, width, channels]
        return:
            style_loss: float
    '''
    vgg_input_bgr = tf.concat([hazy_img, gene_img, clear_img], axis=0)
    f1, f2, f3, f4 = call_vgg_16(vgg_input_bgr)
    hazy_img_f3, gene_img_f3, clear_img_f3 = tf.split(value=f3, num_or_size_splits=3, axis=0)

    style_loss = calc_content_loss(clear_img_f3, gene_img_f3)

    return style_loss


def calc_preceptual_loss_origin(hazy_img, gene_img, clear_img):
    '''calculate preceptual loss

        reference: Perceptual Losses for Real-Time Style Transfer and Super-Resolution

        args:
            hazy_img: [batch_size, height, width, channels]
            gene_img: [batch_size, height, width, channels]
            clear_img: [batch_size, height, width, channels]
        return:
            content_loss: float
            style_loss: float
    '''
    vgg_input_bgr = tf.concat([hazy_img, gene_img, clear_img], axis=0)
    f1, f2, f3, f4 = call_vgg_16(vgg_input_bgr)
    hazy_img_f1, gene_img_f1, clear_img_f1 = tf.split(value=f1, num_or_size_splits=3, axis=0)
    hazy_img_f2, gene_img_f2, clear_img_f2 = tf.split(value=f2, num_or_size_splits=3, axis=0)
    hazy_img_f3, gene_img_f3, clear_img_f3 = tf.split(value=f3, num_or_size_splits=3, axis=0)
    hazy_img_f4, gene_img_f4, clear_img_f4 = tf.split(value=f4, num_or_size_splits=3, axis=0)

    content_loss = calc_content_loss(hazy_img_f3, gene_img_f3)

    style_loss = calc_style_loss(gene_img_f1, clear_img_f1)
    style_loss = style_loss + calc_style_loss(gene_img_f2, clear_img_f2)
    style_loss = style_loss + calc_style_loss(gene_img_f3, clear_img_f3)
    style_loss = style_loss + calc_style_loss(gene_img_f4, clear_img_f4)

    return content_loss, style_loss


def call_vgg_16(input_bgr):
    '''call for vgg_16 model

        args:
            input_bgr: bgr image [batch, height, width, 3] values scaled [-1, 1]
        return:
            f1, f2, f3, f4
    '''
    input_bgr_scaled = data_generator.Data_Generator.de_scale_img(input_bgr)
    input_bgr_224 = tf.image.resize_images(input_bgr_scaled, [224, 224])

    vgg_16_model = vgg_16.Vgg16(vgg16_npy_path=flags.FLAGS.vgg_model_dir)
    vgg_16_model.build(input_bgr_224)

    f1, f2, f3, f4 = vgg_16_model.conv1_2, vgg_16_model.conv2_2, vgg_16_model.conv3_3, vgg_16_model.conv4_3

    return f1, f2, f3, f4


def calc_content_loss(hazy_img_f, gene_img_f):
    '''calculate content loss

        reference: Perceptual Losses for Real-Time Style Transfer and Super-Resolution

        args:
            hazy_img_f: [batch_size, height, width, channels]
            gene_img_f: [batch_size, height, width, channels]
        return:
            content_loss: float
    '''
    content_loss = tf.nn.l2_loss(hazy_img_f - gene_img_f) / tf.to_float(tf.size(hazy_img_f))
    return content_loss


def calc_style_loss(gene_img_f, clear_img_f):
    '''calculate style loss

        reference: Perceptual Losses for Real-Time Style Transfer and Super-Resolution

        args:
            gene_img_f: [batch_size, height, width, channels]
            clear_img_f: [batch_size, height, width, channels]
        return:
            style_loss: float
    '''
    style_loss = tf.nn.l2_loss(gram_matrix(gene_img_f) - gram_matrix(clear_img_f)) / tf.to_float(tf.size(gene_img_f))
    return style_loss


def gram_matrix(input_img):
    '''calculate Gram matrix

        reference: Perceptual Losses for Real-Time Style Transfer and Super-Resolution

        args:
            input_img: [batch_size, height, width, channels]
        return:
            grams: [batch_size, channels, channels]
    '''
    input_shape = tf.shape(input_img)
    batch_size, height, width, channels = input_shape[0], input_shape[1], input_shape[2], input_shape[3]
    input_img = tf.reshape(input_img, tf.stack([batch_size, -1, channels]))
    grams = tf.matmul(input_img, input_img, transpose_a=True) / tf.to_float(height * width * channels)
    return grams
