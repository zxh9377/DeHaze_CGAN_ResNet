import tensorflow as tf
import os
import cv2
import numpy as np
import data_generator
import model
import flags


# check is time for the task with the frequency
def is_time(step, freq, max_steps):
    return freq > 0 and (step % freq == 0 or step == max_steps - 1)


def train():
    train_dataset = data_generator.Data_Generator()
    net_model = model.Model()

    max_steps = flags.FLAGS.max_steps
    save_freq = flags.FLAGS.save_freq
    val_freq = flags.FLAGS.val_freq
    summary_freq = flags.FLAGS.summary_freq
    log_dir = flags.FLAGS.log_dir
    checkpoint_dir = flags.FLAGS.checkpoint_dir
    if not os.path.isdir(log_dir):
        os.mkdir(log_dir)
        print("make log dir : {}".format(log_dir))
    if not os.path.isdir(checkpoint_dir):
        os.mkdir(checkpoint_dir)
        print("make checkpoint dir : {}".format(checkpoint_dir))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter(log_dir + "/", sess.graph)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)

        print("start training...")
        for step in range(max_steps):
            try:
                input_element = sess.run(train_dataset.input)
                feed_dict = {
                    net_model.input_clear: input_element[0],
                    net_model.input_hazy: input_element[1],
                    net_model.training: True,
                    net_model.keep_prob: flags.FLAGS.keep_prob
                }

                fetches = {
                    "generator_train": net_model.generator_train,
                    "discriminator": net_model.discriminator_train,
                }
                if is_time(step, val_freq, max_steps):
                    fetches["generator_loss"] = net_model.generator_loss
                    fetches["discriminator_loss"] = net_model.discriminator_loss
                if is_time(step, summary_freq, max_steps):
                    fetches["merged"] = merged

                fetches_result = sess.run(fetches, feed_dict=feed_dict)

                if is_time(step, val_freq, max_steps):
                    print("step {0} : discrimator_loss:{1}\tgenenrator_loss:{2}"
                          .format(step, fetches_result["discriminator_loss"], fetches_result["generator_loss"]))
                if is_time(step, summary_freq, max_steps):
                    writer.add_summary(fetches_result["merged"], step)

                if is_time(step, save_freq, max_steps):
                    saver.save(sess, os.path.join(checkpoint_dir, "dehaze-model"), global_step=step)
                    print("save checkpoint at step {}".format(step))

            except tf.errors.OutOfRangeError:
                break
        print("training done...")


def use():
    use_dataset = data_generator.Data_Generator()
    net_model = model.Model()

    checkpoint_dir = flags.FLAGS.checkpoint_dir
    use_store_dir = flags.FLAGS.use_store_dir
    output_filetype = flags.FLAGS.output_filetype

    store_images = []

    with tf.Session() as sess:
        # sess.run(tf.global_variables_initializer())

        saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)
        ckpt = tf.train.latest_checkpoint(checkpoint_dir)
        if ckpt:
            saver.restore(sess, ckpt)
            print("restore model from checkpoint {}...".format(ckpt))
        else:
            raise Exception("Message: no checkpoint be found!")

        print("start run model...")
        while (True):
            try:
                input_element = sess.run(use_dataset.input)
                feed_dict = {
                    net_model.input_hazy: input_element,
                    net_model.keep_prob: 1.0
                }

                fetches = {
                    "generator_output": net_model.generator_output
                }

                fetches_result = sess.run(fetches, feed_dict=feed_dict)

                for hazy, gene in zip(input_element, fetches_result["generator_output"]):
                    store_images.append((hazy, gene))

            except tf.errors.OutOfRangeError:
                print("run model over...")
                break

    print("start store output images...")
    for i, imgs in enumerate(store_images):
        hazy, gene = imgs[0], imgs[1]
        hazy = data_generator.Data_Generator.de_scale_img(hazy)  # scale image [-1,1] => [0,255]
        gene = data_generator.Data_Generator.de_scale_img(gene)
        img = np.concatenate((hazy, gene), axis=1)
        cv2.imwrite(os.path.join(use_store_dir, "{0}.{1}".format(i, output_filetype)), img)
    print("store output done...")


if __name__ == "__main__":
    if flags.FLAGS.mode == "train":
        train()
    if flags.FLAGS.mode == "use":
        use()
