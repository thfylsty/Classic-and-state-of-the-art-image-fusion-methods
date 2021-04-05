# -*- coding: utf-8 -*-
from utils import (
  read_data, 
  input_setup, 
  imsave,
  merge,
  gradient,
  lrelu,
  weights_spectral_norm,
  l2_norm
)

import time
import os
import matplotlib.pyplot as plt

import numpy as np
import tensorflow as tf

class CGAN(object):

  def __init__(self, 
               sess, 
               image_size=132,
               label_size=120,
               batch_size=32,
               c_dim=1, 
               checkpoint_dir=None, 
               sample_dir=None):

    self.sess = sess
    self.is_grayscale = (c_dim == 1)
    self.image_size = image_size
    self.label_size = label_size
    self.batch_size = batch_size

    self.c_dim = c_dim

    self.checkpoint_dir = checkpoint_dir
    self.sample_dir = sample_dir
    self.build_model()

  def build_model(self):
    with tf.name_scope('IR_input'):
        #红外图像patch
        self.images_ir = tf.placeholder(tf.float32, [None, self.image_size, self.image_size, self.c_dim], name='images_ir')
        self.labels_ir = tf.placeholder(tf.float32, [None, self.label_size, self.label_size, self.c_dim], name='labels_ir')
    with tf.name_scope('VI_input'):
        #可见光图像patch
        self.images_vi = tf.placeholder(tf.float32, [None, self.image_size, self.image_size, self.c_dim], name='images_vi')
        self.labels_vi = tf.placeholder(tf.float32, [None, self.label_size, self.label_size, self.c_dim], name='labels_vi')
        #self.labels_vi_gradient=gradient(self.labels_vi)
    #将红外和可见光图像在通道方向连起来，第一通道是红外图像，第二通道是可见光图像
    with tf.name_scope('input'):
        #self.resize_ir=tf.image.resize_images(self.images_ir, (self.image_size, self.image_size), method=2)
        self.input_image_ir =tf.concat([self.labels_ir,self.labels_ir,self.labels_vi],axis=-1)
        self.input_image_vi =tf.concat([self.labels_vi,self.labels_vi,self.labels_ir],axis=-1)
    #self.pred=tf.clip_by_value(tf.sign(self.pred_ir-self.pred_vi),0,1)
    #融合图像
    with tf.name_scope('fusion'): 
        self.fusion_image=self.fusion_model(self.input_image_ir,self.input_image_vi)

    with tf.name_scope('g_loss'):
        #self.g_loss_1=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=neg, labels=tf.ones_like(neg)))
        #self.g_loss_1=tf.reduce_mean(tf.square(neg-tf.ones_like(pos)))
        #self.g_loss_1=tf.reduce_mean(tf.square(neg-tf.random_uniform(shape=[self.batch_size,1],minval=0.7,maxval=1.2,dtype=tf.float32)))
        #tf.summary.scalar('g_loss_1',self.g_loss_1)
        #self.g_loss_2=tf.reduce_mean(tf.square(self.fusion_image - self.labels_ir))
        self.g_loss_2=1.5*tf.reduce_mean(tf.square(self.fusion_image - self.labels_ir))+1.5*tf.reduce_mean(tf.square(self.fusion_image - self.labels_vi))+16*tf.reduce_mean(tf.square(gradient(self.fusion_image) -gradient (self.labels_vi)))+16*tf.reduce_mean(tf.square(gradient(self.fusion_image) -gradient (self.labels_ir)))
        tf.summary.scalar('g_loss_2',self.g_loss_2)

        self.g_loss_total=100*self.g_loss_2
        tf.summary.scalar('loss_g',self.g_loss_total)
    self.saver = tf.train.Saver(max_to_keep=50)





    with tf.name_scope('image'):
        tf.summary.image('input_ir',tf.expand_dims(self.labels_ir[1,:,:,:],0))  
        tf.summary.image('input_vi',tf.expand_dims(self.labels_vi[1,:,:,:],0))  
        tf.summary.image('fusion_image',tf.expand_dims(self.fusion_image[1,:,:,:],0))   
    
  def train(self, config):
    if config.is_train:
      input_setup(self.sess, config,"Train_ir")
      input_setup(self.sess,config,"Train_vi")
    else:
      nx_ir, ny_ir = input_setup(self.sess, config,"Test_ir")
      nx_vi,ny_vi=input_setup(self.sess, config,"Test_vi")

    if config.is_train:     
      data_dir_ir = os.path.join('./{}'.format(config.checkpoint_dir), "Train_ir","train.h5")
      data_dir_vi = os.path.join('./{}'.format(config.checkpoint_dir), "Train_vi","train.h5")
    else:
      data_dir_ir = os.path.join('./{}'.format(config.checkpoint_dir),"Test_ir", "test.h5")
      data_dir_vi = os.path.join('./{}'.format(config.checkpoint_dir),"Test_vi", "test.h5")

    train_data_ir, train_label_ir = read_data(data_dir_ir)
    train_data_vi, train_label_vi = read_data(data_dir_vi)
    #找训练时更新的变量组（判决器和生成器是分开训练的，所以要找到对应的变量）
    t_vars = tf.trainable_variables()
    self.d_vars = [var for var in t_vars if 'discriminator' in var.name]
    print(self.d_vars)
    self.g_vars = [var for var in t_vars if 'fusion_model' in var.name]
    print(self.g_vars)
    # clip_ops = []
    # for var in self.d_vars:
        # clip_bounds = [-.01, .01]
        # clip_ops.append(
            # tf.assign(
                # var, 
                # tf.clip_by_value(var, clip_bounds[0], clip_bounds[1])
            # )
        # )
    # self.clip_disc_weights = tf.group(*clip_ops)
    # Stochastic gradient descent with the standard backpropagation
    with tf.name_scope('train_step'):
        self.train_fusion_op = tf.train.AdamOptimizer(config.learning_rate).minimize(self.g_loss_total,var_list=self.g_vars)
        #self.train_discriminator_op=tf.train.AdamOptimizer(config.learning_rate).minimize(self.d_loss,var_list=self.d_vars)
    #将所有统计的量合起来
    self.summary_op = tf.summary.merge_all()
    #生成日志文件
    self.train_writer = tf.summary.FileWriter(config.summary_dir + '/train',self.sess.graph,flush_secs=60)
    
    tf.initialize_all_variables().run()
    
    counter = 0
    start_time = time.time()

    # if self.load(self.checkpoint_dir):
      # print(" [*] Load SUCCESS")
    # else:
      # print(" [!] Load failed...")

    if config.is_train:
      print("Training...")

      for ep in xrange(config.epoch):
        # Run by batch images
        batch_idxs = len(train_data_ir) // config.batch_size
        for idx in xrange(0, batch_idxs):
          batch_images_ir = train_data_ir[idx*config.batch_size : (idx+1)*config.batch_size]
          batch_labels_ir = train_label_ir[idx*config.batch_size : (idx+1)*config.batch_size]
          batch_images_vi = train_data_vi[idx*config.batch_size : (idx+1)*config.batch_size]
          batch_labels_vi = train_label_vi[idx*config.batch_size : (idx+1)*config.batch_size]

          counter += 1
          #for i in range(2):
           # _, err_d= self.sess.run([self.train_discriminator_op, self.d_loss], feed_dict={self.images_ir: batch_images_ir, self.images_vi: batch_images_vi, self.labels_vi: batch_labels_vi,self.labels_ir:batch_labels_ir})
            # self.sess.run(self.clip_disc_weights)
          _, err_g,summary_str= self.sess.run([self.train_fusion_op, self.g_loss_total,self.summary_op], feed_dict={self.images_ir: batch_images_ir, self.images_vi: batch_images_vi, self.labels_ir: batch_labels_ir,self.labels_vi:batch_labels_vi})
          #将统计的量写到日志文件里
          self.train_writer.add_summary(summary_str,counter)

          if counter % 10 == 0:
            print("Epoch: [%2d], step: [%2d], time: [%4.4f], loss_g:[%.8f]" \
              % ((ep+1), counter, time.time()-start_time, err_g))
            #print(a)

        self.save(config.checkpoint_dir, ep)

    else:
      print("Testing...")

      result = self.fusion_image.eval(feed_dict={self.images_ir: train_data_ir, self.labels_ir: train_label_ir,self.images_vi: train_data_vi, self.labels_vi: train_label_vi})
      result=result*127.5+127.5
      result = merge(result, [nx_ir, ny_ir])
      result = result.squeeze()
      image_path = os.path.join(os.getcwd(), config.sample_dir)
      image_path = os.path.join(image_path, "test_image.png")
      imsave(result, image_path)

  def fusion_model(self,img_ir,img_vi):
####################  Layer1  ###########################
    with tf.variable_scope('fusion_model'):
        with tf.variable_scope('layer1'):
            weights=tf.get_variable("w1",[5,5,3,16],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            bias=tf.get_variable("b1",[16],initializer=tf.constant_initializer(0.0))
            conv1_ir= tf.contrib.layers.batch_norm(tf.nn.conv2d(img_ir, weights, strides=[1,1,1,1], padding='SAME') + bias, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True)
            conv1_ir = lrelu(conv1_ir)   
        with tf.variable_scope('layer1_vi'):
            weights=tf.get_variable("w1_vi",[5,5,3,16],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            bias=tf.get_variable("b1_vi",[16],initializer=tf.constant_initializer(0.0))
            conv1_vi= tf.contrib.layers.batch_norm(tf.nn.conv2d(img_vi, weights, strides=[1,1,1,1], padding='SAME') + bias, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True)
            conv1_vi = lrelu(conv1_vi)           
            

####################  Layer2  ###########################            
        with tf.variable_scope('layer2'):
            weights=tf.get_variable("w2",[3,3,16,16],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            bias=tf.get_variable("b2",[16],initializer=tf.constant_initializer(0.0))
            conv2_ir= tf.contrib.layers.batch_norm(tf.nn.conv2d(conv1_ir, weights, strides=[1,1,1,1], padding='SAME') + bias, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True)
            conv2_ir = lrelu(conv2_ir)         
            
        with tf.variable_scope('layer2_vi'):
            weights=tf.get_variable("w2_vi",[3,3,16,16],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            bias=tf.get_variable("b2_vi",[16],initializer=tf.constant_initializer(0.0))
            conv2_vi= tf.contrib.layers.batch_norm(tf.nn.conv2d(conv1_vi, weights, strides=[1,1,1,1], padding='SAME') + bias, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True)
            conv2_vi = lrelu(conv2_vi)            
            

        conv_2_midle =tf.concat([conv2_ir,conv2_vi],axis=-1)    
       
  
        with tf.variable_scope('layer2_3'):
            weights=tf.get_variable("w2_3",[1,1,32,16],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            bias=tf.get_variable("b2_3",[16],initializer=tf.constant_initializer(0.0))
            conv2_3_ir= tf.contrib.layers.batch_norm(tf.nn.conv2d(conv_2_midle, weights, strides=[1,1,1,1], padding='SAME') + bias, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True)
            conv2_3_ir = lrelu(conv2_3_ir)   
                    
                       
        with tf.variable_scope('layer2_3_vi'):
            weights=tf.get_variable("w2_3_vi",[1,1,32,16],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            bias=tf.get_variable("b2_3_vi",[16],initializer=tf.constant_initializer(0.0))
            conv2_3_vi= tf.contrib.layers.batch_norm(tf.nn.conv2d(conv_2_midle, weights, strides=[1,1,1,1], padding='SAME') + bias, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True)
            conv2_3_vi = lrelu(conv2_3_vi)       
            
            
####################  Layer3  ###########################               
        conv_12_ir=tf.concat([conv1_ir,conv2_ir,conv2_3_ir],axis=-1)
        conv_12_vi=tf.concat([conv1_vi,conv2_vi,conv2_3_vi],axis=-1)        
            
        with tf.variable_scope('layer3'):
            weights=tf.get_variable("w3",[3,3,48,16],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            bias=tf.get_variable("b3",[16],initializer=tf.constant_initializer(0.0))
            conv3_ir= tf.contrib.layers.batch_norm(tf.nn.conv2d(conv_12_ir, weights, strides=[1,1,1,1], padding='SAME') + bias, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True)
            conv3_ir =lrelu(conv3_ir)
        with tf.variable_scope('layer3_vi'):
            weights=tf.get_variable("w3_vi",[3,3,48,16],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            bias=tf.get_variable("b3_vi",[16],initializer=tf.constant_initializer(0.0))
            conv3_vi= tf.contrib.layers.batch_norm(tf.nn.conv2d(conv_12_vi, weights, strides=[1,1,1,1], padding='SAME') + bias, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True)
            conv3_vi = lrelu(conv3_vi)
            

        conv_3_midle =tf.concat([conv3_ir,conv3_vi],axis=-1)    
       
  
        with tf.variable_scope('layer3_4'):
            weights=tf.get_variable("w3_4",[1,1,32,16],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            bias=tf.get_variable("b3_4",[16],initializer=tf.constant_initializer(0.0))
            conv3_4_ir= tf.contrib.layers.batch_norm(tf.nn.conv2d(conv_3_midle, weights, strides=[1,1,1,1], padding='SAME') + bias, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True)
            conv3_4_ir = lrelu(conv3_4_ir)   
                    
                       
        with tf.variable_scope('layer3_4_vi'):
            weights=tf.get_variable("w3_4_vi",[1,1,32,16],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            bias=tf.get_variable("b3_4_vi",[16],initializer=tf.constant_initializer(0.0))
            conv3_4_vi= tf.contrib.layers.batch_norm(tf.nn.conv2d(conv_3_midle, weights, strides=[1,1,1,1], padding='SAME') + bias, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True)
            conv3_4_vi = lrelu(conv3_4_vi)  


            
####################  Layer4  ########################### 
        conv_123_ir=tf.concat([conv1_ir,conv2_ir,conv3_ir,conv3_4_ir],axis=-1)
        conv_123_vi=tf.concat([conv1_vi,conv2_vi,conv3_vi,conv3_4_vi],axis=-1)                   
            
        with tf.variable_scope('layer4'):
            weights=tf.get_variable("w4",[3,3,64,16],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            bias=tf.get_variable("b4",[16],initializer=tf.constant_initializer(0.0))
            conv4_ir= tf.contrib.layers.batch_norm(tf.nn.conv2d(conv_123_ir, weights, strides=[1,1,1,1], padding='SAME') + bias, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True)
            conv4_ir = lrelu(conv4_ir)
        with tf.variable_scope('layer4_vi'):
            weights=tf.get_variable("w4_vi",[3,3,64,16],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            bias=tf.get_variable("b4_vi",[16],initializer=tf.constant_initializer(0.0))
            conv4_vi= tf.contrib.layers.batch_norm(tf.nn.conv2d(conv_123_vi, weights, strides=[1,1,1,1], padding='SAME') + bias, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True)
            conv4_vi = lrelu(conv4_vi)
            
 
        conv_ir_vi =tf.concat([conv1_ir,conv1_vi,conv2_ir,conv2_vi,conv3_ir,conv3_vi,conv4_ir,conv4_vi],axis=-1)
 
        
        with tf.variable_scope('layer5'):
            weights=tf.get_variable("w5",[1,1,128,1],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            bias=tf.get_variable("b5",[1],initializer=tf.constant_initializer(0.0))
            conv5_ir= tf.nn.conv2d(conv_ir_vi, weights, strides=[1,1,1,1], padding='SAME') + bias
            conv5_ir=tf.nn.tanh(conv5_ir)
    return conv5_ir
    
  '''  
  def discriminator(self,img,reuse,update_collection=None):
    with tf.variable_scope('discriminator',reuse=reuse):
        print(img.shape)
        with tf.variable_scope('layer_1'):
            weights=tf.get_variable("w_1",[3,3,1,32],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            weights=weights_spectral_norm(weights,update_collection=update_collection)
            bias=tf.get_variable("b_1",[32],initializer=tf.constant_initializer(0.0))
            conv1_vi=tf.nn.conv2d(img, weights, strides=[1,2,2,1], padding='VALID') + bias
            conv1_vi = lrelu(conv1_vi)
            #print(conv1_vi.shape)
        with tf.variable_scope('layer_2'):
            weights=tf.get_variable("w_2",[3,3,32,64],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            weights=weights_spectral_norm(weights,update_collection=update_collection)
            bias=tf.get_variable("b_2",[64],initializer=tf.constant_initializer(0.0))
            conv2_vi= tf.contrib.layers.batch_norm(tf.nn.conv2d(conv1_vi, weights, strides=[1,2,2,1], padding='VALID') + bias, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True)
            conv2_vi = lrelu(conv2_vi)
            #print(conv2_vi.shape)
        with tf.variable_scope('layer_3'):
            weights=tf.get_variable("w_3",[3,3,64,128],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            weights=weights_spectral_norm(weights,update_collection=update_collection)
            bias=tf.get_variable("b_3",[128],initializer=tf.constant_initializer(0.0))
            conv3_vi= tf.contrib.layers.batch_norm(tf.nn.conv2d(conv2_vi, weights, strides=[1,2,2,1], padding='VALID') + bias, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True)
            conv3_vi=lrelu(conv3_vi)
            #print(conv3_vi.shape)
        with tf.variable_scope('layer_4'):
            weights=tf.get_variable("w_4",[3,3,128,256],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            weights=weights_spectral_norm(weights,update_collection=update_collection)
            bias=tf.get_variable("b_4",[256],initializer=tf.constant_initializer(0.0))
            conv4_vi= tf.contrib.layers.batch_norm(tf.nn.conv2d(conv3_vi, weights, strides=[1,2,2,1], padding='VALID') + bias, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True)
            conv4_vi=lrelu(conv4_vi)
            conv4_vi = tf.reshape(conv4_vi,[self.batch_size,6*6*256])
        with tf.variable_scope('line_5'):
            weights=tf.get_variable("w_5",[6*6*256,2],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            weights=weights_spectral_norm(weights,update_collection=update_collection)
            bias=tf.get_variable("b_5",[2],initializer=tf.constant_initializer(0.0))
            line_5=tf.matmul(conv4_vi, weights) + bias
            #conv3_vi= tf.contrib.layers.batch_norm(conv3_vi, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True)
    return line_5
    '''
  def save(self, checkpoint_dir, step):
    model_name = "CGAN.model"
    model_dir = "%s_%s" % ("CGAN", self.label_size)
    checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    self.saver.save(self.sess,
                    os.path.join(checkpoint_dir, model_name),
                    global_step=step)

  def load(self, checkpoint_dir):
    print(" [*] Reading checkpoints...")
    model_dir = "%s_%s" % ("CGAN", self.label_size)
    checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        print(ckpt_name)
        self.saver.restore(self.sess, os.path.join(checkpoint_dir,ckpt_name))
        return True
    else:
        return False
