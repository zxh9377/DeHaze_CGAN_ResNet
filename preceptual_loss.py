import tensorflow as tf
import vgg.vgg_simple as vgg
import flags


def calc_preceptual_loss(hazy_img, gene_img, clear_img):
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
    f1, f2, f3, f4, exclude = vgg.vgg_16(tf.concat([hazy_img, gene_img, clear_img], axis=0))
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
    # input_img = tf.reshape(input_img, [flags.FLAGS.batch_size, -1, flags.FLAGS.img_channels])
    input_img = tf.reshape(input_img, tf.stack([batch_size, -1, channels]))
    grams = tf.matmul(input_img, input_img, transpose_a=True) / tf.to_float(height * width * channels)
    return grams
