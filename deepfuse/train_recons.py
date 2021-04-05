# Train the DeepFuse Net

from __future__ import print_function

import scipy.io as scio
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from ssim_loss_function import SSIM_LOSS
from deep_fuse_net import DeepFuseNet
from utils import get_train_images

STYLE_LAYERS  = ('relu1_1', 'relu2_1', 'relu3_1', 'relu4_1')

TRAINING_IMAGE_SHAPE = (256, 256, 1) # (height, width, color_channels)
TRAINING_IMAGE_SHAPE_OR = (256, 256, 1) # (height, width, color_channels)

LEARNING_RATE = 1e-4
EPSILON = 1e-5

def train_recons(original_imgs_path, save_path, model_pre_path, EPOCHES_set, BATCH_SIZE, debug=False, logging_period=100):
    if debug:
        from datetime import datetime
        start_time = datetime.now()
    EPOCHS = EPOCHES_set
    print("EPOCHES   : ", EPOCHS)
    print("BATCH_SIZE: ", BATCH_SIZE)
    num_imgs = len(original_imgs_path)
    # num_imgs = 100
    original_imgs_path = original_imgs_path[:num_imgs]
    mod = num_imgs % BATCH_SIZE

    print('Train images number %d.\n' % num_imgs)
    print('Train images samples %s.\n' % str(num_imgs / BATCH_SIZE))

    if mod > 0:
        print('Train set has been trimmed %d samples...\n' % mod)
        original_imgs_path = original_imgs_path[:-mod]

    # get the traing image shape
    HEIGHT, WIDTH, CHANNELS = TRAINING_IMAGE_SHAPE
    INPUT_SHAPE = (BATCH_SIZE, HEIGHT, WIDTH, CHANNELS)

    HEIGHT_OR, WIDTH_OR, CHANNELS_OR = TRAINING_IMAGE_SHAPE_OR
    INPUT_SHAPE_OR = (BATCH_SIZE, HEIGHT_OR, WIDTH_OR, CHANNELS_OR)

    # create the graph
    with tf.Graph().as_default(), tf.Session() as sess:
        original = tf.placeholder(tf.float32, shape=INPUT_SHAPE_OR, name='original')
        source = original

        print('source  :', source.shape)
        print('original:', original.shape)

        # create the deepfuse net (encoder and decoder)
        dfn = DeepFuseNet(model_pre_path)

        generated_img = dfn.transform_recons(source)

        print('generate:', generated_img.shape)

        ssim_loss = SSIM_LOSS(original, generated_img)

        loss = 1-ssim_loss
        train_op = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)

        sess.run(tf.global_variables_initializer())

        # saver = tf.train.Saver()
        saver = tf.train.Saver(keep_checkpoint_every_n_hours=1)

        # ** Start Training **
        step = 0
        count_loss = 0
        n_batches = int(len(original_imgs_path) // BATCH_SIZE)

        if debug:
            elapsed_time = datetime.now() - start_time
            print('\nElapsed time for preprocessing before actually train the model: %s' % elapsed_time)
            print('Now begin to train the model...\n')
            start_time = datetime.now()

        Loss_all = [i for i in range(EPOCHS * n_batches)]
        for epoch in range(EPOCHS):

            np.random.shuffle(original_imgs_path)

            for batch in range(n_batches):
                # retrive a batch of content and style images

                original_path = original_imgs_path[batch*BATCH_SIZE:(batch*BATCH_SIZE + BATCH_SIZE)]
                original_batch = get_train_images(original_path, crop_height=HEIGHT, crop_width=WIDTH, flag=False)
                original_batch = original_batch.reshape([BATCH_SIZE, 256, 256, 1])

                # print('original_batch shape final:', original_batch.shape)

                # run the training step
                sess.run(train_op, feed_dict={original: original_batch})
                step += 1
                # if step % 1000 == 0:
                #     saver.save(sess, save_path, global_step=step)
                if debug:
                    is_last_step = (epoch == EPOCHS - 1) and (batch == n_batches - 1)

                    if is_last_step or step % logging_period == 0:
                        elapsed_time = datetime.now() - start_time
                        _ssim_loss, _loss = sess.run([ssim_loss, loss], feed_dict={original: original_batch})
                        Loss_all[count_loss] = _loss
                        count_loss += 1
                        print('Deep fuse==>>step: %d,  total loss: %s,  elapsed time: %s' % (step, _loss, elapsed_time))
                        print('ssim_loss: %s ' % (_ssim_loss))

        # ** Done Training & Save the model **
        saver.save(sess, save_path)

        loss_data = Loss_all[:count_loss]
        scio.savemat('D:/project/GitHub/ImageFusion/Imagefusion_deepfuse/DeepFuseLossData.mat', {'loss': loss_data})

        # iter_index = [i for i in range(count_loss)]
        # plt.plot(iter_index, Loss_all[:count_loss])
        # plt.show()

        if debug:
            elapsed_time = datetime.now() - start_time
            print('Done training! Elapsed time: %s' % elapsed_time)
            print('Model is saved to: %s' % save_path)

