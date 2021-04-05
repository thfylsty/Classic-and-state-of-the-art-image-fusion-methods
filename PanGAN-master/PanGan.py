import tensorflow as tf
import numpy as np
import ops as Op

class PanGan(object):
    
    def __init__(self, pan_size, ms_size, batch_size,num_spectrum, ratio,init_lr=0.001,lr_decay_rate=0.99,lr_decay_step=1000, is_training=True):
        
        self.num_spectrum=num_spectrum
        self.is_training=is_training
        self.ratio = ratio
        self.batch_size=batch_size
        self.pan_size=pan_size
        self.ms_size=ms_size
        self.init_lr=init_lr
        self.lr_decay_rate=lr_decay_rate
        self.lr_decay_step=lr_decay_step
        self.build_model(pan_size, ms_size, batch_size,num_spectrum, is_training)
        
    def build_model(self, pan_size, ms_size, batch_size, num_spectrum, is_training):
        
        if is_training:
            with tf.name_scope('input'):
                self.pan_img=tf.placeholder(dtype=tf.float32, shape=(batch_size, pan_size, pan_size, 1), name='pan_placeholder')
                self.ms_img=tf.placeholder(dtype=tf.float32, shape=(batch_size,ms_size, ms_size, num_spectrum), name='ms_placeholder')
                self.ms_img_=tf.image.resize_images(images=self.ms_img, size=[pan_size, pan_size],
                                                               method=2)
                self.pan_img_hp=self.high_pass_2(self.pan_img, 'pan')##2019-11-17
                self.pan_img_4=self.AddChannel(self.pan_img)
                #self.ms_img_hp=self.high_pass(self.ms_img, 'ms')
            with tf.name_scope('PanSharpening'):
                self.PanSharpening_img= self.PanSharpening_model_dense(self.pan_img, self.ms_img)
                #self.PanSharpening_img_blur=self.blur(self.PanSharpening_img,7)
                self.PanSharpening_img_=tf.image.resize_images(images=self.PanSharpening_img, size=[ms_size, ms_size],
                                                               method=tf.image.ResizeMethod.BILINEAR)

                self.PanSharpening_img_pan=tf.reshape(tf.reduce_mean(self.PanSharpening_img, axis=3), (batch_size, pan_size, pan_size, 1))
                #self.PanSharpening_img_hp=self.high_pass_1(self.PanSharpening_img)11-17

                self.PanSharpening_img_hp = self.high_pass_2(self.PanSharpening_img_pan)##2019-11-17
            
            with tf.name_scope('d_loss'):
                with tf.name_scope('spatial_loss'):
                    #spatial_pos=self.spatial_discriminator(self.pan_img_4, reuse=False)11-17
                    spatial_pos = self.spatial_discriminator(self.pan_img, reuse=False)
                    #spatial_neg=self.spatial_discriminator(self.PanSharpening_img, reuse=True)11-17
                    spatial_neg = self.spatial_discriminator(self.PanSharpening_img_pan, reuse=True)
                    # spatial_pos_loss= tf.reduce_mean(tf.square(spatial_pos-tf.random_uniform(shape=[self.batch_size,1],minval=0.7,maxval=1.2)))
                    # spatial_neg_loss= tf.reduce_mean(tf.square(spatial_neg-tf.random_uniform(shape=[self.batch_size,1],minval=0,maxval=0.3)))
                    spatial_pos_loss= tf.reduce_mean(tf.square(spatial_pos-tf.ones(shape=[batch_size,1], dtype=tf.float32)))
                    spatial_neg_loss= tf.reduce_mean(tf.square(spatial_neg-tf.zeros(shape=[batch_size,1], dtype=tf.float32)))
                    self.spatial_loss=spatial_pos_loss + spatial_neg_loss
                    tf.summary.scalar('spatial_loss', self.spatial_loss)
                with tf.name_scope('spectrum_loss'):
                    spectrum_pos=self.spectrum_discriminator(self.ms_img_, reuse=False)
                    spectrum_neg=self.spectrum_discriminator(self.PanSharpening_img, reuse=True)
                    # spectrum_pos_loss= tf.reduce_mean(tf.square(spectrum_pos-tf.random_uniform(shape=[self.batch_size,1],minval=0.7,maxval=1.2)))
                    # spectrum_neg_loss= tf.reduce_mean(tf.square(spectrum_neg-tf.random_uniform(shape=[self.batch_size,1],minval=0,maxval=0.3)))
                    spectrum_pos_loss= tf.reduce_mean(tf.square(spectrum_pos-tf.ones(shape=[batch_size,1], dtype=tf.float32)))
                    spectrum_neg_loss= tf.reduce_mean(tf.square(spectrum_neg-tf.zeros(shape=[batch_size,1], dtype=tf.float32)))
                    self.spectrum_loss=spectrum_pos_loss + spectrum_neg_loss
                    tf.summary.scalar('spectrum_loss', self.spectrum_loss)
            
            with tf.name_scope('g_loss'):
                spatial_loss_ad= tf.reduce_mean(tf.square(spatial_neg-tf.ones(shape=[batch_size,1], dtype=tf.float32)))
                tf.summary.scalar('spatial_loss_ad', spatial_loss_ad)
                spectrum_loss_ad=tf.reduce_mean(tf.square(spectrum_neg-tf.ones(shape=[batch_size,1], dtype=tf.float32)))
                tf.summary.scalar('spectrum_loss_ad', spectrum_loss_ad)
                g_spatital_loss= tf.reduce_mean(tf.square(self.PanSharpening_img_hp-self.pan_img_hp))
                #g_spatital_loss= tf.reduce_mean(tf.reduce_sum(tf.sqrt(tf.reduce_sum(tf.square(self.PanSharpening_img_hp-self.pan_img_hp),2)),1))
                tf.summary.scalar('g_spatital_loss', g_spatital_loss)
                g_spectrum_loss=tf.reduce_mean(tf.square(self.PanSharpening_img-self.ms_img_))####2019-11-19 改成大size
                #g_spectrum_loss=0.5*tf.reduce_mean(tf.reduce_sum(tf.square(self.PanSharpening_img_-self.ms_img),[2,3]))
                tf.summary.scalar('g_spectrum_loss', g_spectrum_loss)
                self.g_loss= 5*spatial_loss_ad + spectrum_loss_ad+5*g_spatital_loss + g_spectrum_loss
                #self.g_loss= (spectrum_loss_ad+spatial_loss_ad) + 1000*(g_spatital_loss + 5*g_spectrum_loss)
                #self.g_loss=g_spatital_loss+5*g_spectrum_loss
                #self.g_loss=500*(5*g_spatital_loss + g_spectrum_loss) + 0.5*spatial_loss_ad
                #self.g_loss=500*(g_spatital_loss+g_spectrum_loss)#+spectrum_loss_ad#+spatial_loss_ad++#2019-11-17
                tf.summary.scalar('g_loss', self.g_loss)
                
            # with tf.name_scope('valid_error'):
                # self.valid_spatital_error=tf.reduce_mean(tf.abs(self.PanSharpening_img_hp-self.pan_img_hp))
                # tf.summary.scalar('valid_spatital_error', self.valid_spatital_error)
                # self.valid_spectrum_error=tf.reduce_mean(tf.abs(self.PanSharpening_img-self.ms_img_))
                # tf.summary.scalar('valid_spectrum_error', self.valid_spectrum_error)
        else:
            with tf.name_scope('input'):
                # self.pan_img=tf.placeholder(dtype=tf.float32, shape=(batch_size, pan_size, pan_size, 1), name='pan_placeholder')
                # self.ms_img=tf.placeholder(dtype=tf.float32, shape=(batch_size,ms_size, ms_size, num_spectrum), name='ms_placeholder')
                self.pan_img=tf.placeholder(dtype=tf.float32, shape=(batch_size,128, 128, 1), name='pan_placeholder')
                self.ms_img=tf.placeholder(dtype=tf.float32, shape=(batch_size,32, 32, num_spectrum), name='ms_placeholder')
            self.PanSharpening_img=self.PanSharpening_model_dense(self.pan_img, self.ms_img)
            #self.PanSharpening_img_blur=self.blur(self.PanSharpening_img,7)
            self.PanSharpening_img_=tf.image.resize_images(images=self.PanSharpening_img, size=[32, 32],
                                                               method=tf.image.ResizeMethod.BILINEAR)
            PanSharpening_img_hp=self.high_pass_1(self.PanSharpening_img)
            pan_img_hp=self.high_pass_1(self.pan_img, 'pan')
            self.g_spectrum_loss=tf.reduce_mean(tf.square(self.PanSharpening_img_-self.ms_img))
            self.g_spatial_loss=tf.reduce_mean(tf.square(PanSharpening_img_hp-pan_img_hp))

    def train(self):
        t_vars = tf.trainable_variables()
        d_spatial_vars = [var for var in t_vars if 'spatial_discriminator' in var.name]
        d_spectrum_vars=[var for var in t_vars if 'spectrum_discriminator' in var.name]
        g_vars = [var for var in t_vars if 'Pan_model' in var.name]
        with tf.name_scope('train_step'):
            self.global_step=tf.contrib.framework.get_or_create_global_step()
            self.learning_rate=tf.train.exponential_decay(self.init_lr, global_step=self.global_step, decay_rate=self.lr_decay_rate,
                                                          decay_steps=self.lr_decay_step)
            tf.summary.scalar('global learning rate', self.learning_rate)
            self.train_Pan_model=tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.g_loss, var_list=g_vars, global_step=self.global_step)
            self.train_spatial_discrim=tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.spatial_loss, var_list=d_spatial_vars)
            self.train_spectrum_discrim=tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.spectrum_loss, var_list=d_spectrum_vars)


    def PanSharpening_model_dense(self,pan_img, ms_img):
        with tf.variable_scope('Pan_model'):
            #if self.is_training:
            if True:
                with tf.name_scope('upscale'):
                    # de_weight=tf.get_variable('de_weight', [3,3,self.num_spectrum, self.num_spectrum],
                                            # initializer=tf.truncated_normal_initializer(stddev=1e-3) )
                    # ms_scale4 = tf.nn.conv2d_transpose(ms_img, de_weight, output_shape=[self.batch_size,self.pan_size,self.pan_size,self.num_spectrum],
                                                       # strides=[1,4,4,1],padding='SAME' )                            
                    ms_img=tf.image.resize_images(ms_img, [128, 128], method=2)
            input=tf.concat([ms_img,pan_img],axis=-1)
            with tf.variable_scope('layer1'):
                weights = tf.get_variable("w1", [9, 9, self.num_spectrum + 1, 64],
                                          initializer=tf.truncated_normal_initializer(stddev=1e-3))
                bias = tf.get_variable("b1", [64], initializer=tf.constant_initializer(0.0))
                conv1 = tf.contrib.layers.batch_norm(tf.nn.conv2d(input, weights, strides=[1, 1, 1, 1], padding='SAME') + bias,
                                                     decay=0.9,updates_collections=None, epsilon=1e-5, scale=True)
                conv1= tf.nn.relu(conv1)
            with tf.variable_scope('layer2'):
                weights = tf.get_variable("w2", [5, 5, 64+5, 32],
                                          initializer=tf.truncated_normal_initializer(stddev=1e-3))
                bias = tf.get_variable("b1", [32], initializer=tf.constant_initializer(0.0))
                conv2 = tf.contrib.layers.batch_norm(tf.nn.conv2d(tf.concat([input,conv1],-1), weights, strides=[1, 1, 1, 1], padding='SAME') + bias,
                                                     decay=0.9,updates_collections=None, epsilon=1e-5, scale=True)
                conv2= tf.nn.relu(conv2)
            with tf.variable_scope('layer3'):
                weights = tf.get_variable("w3", [5, 5, 5+64+32 , self.num_spectrum],
                                          initializer=tf.truncated_normal_initializer(stddev=1e-3))
                bias = tf.get_variable("b3", [self.num_spectrum], initializer=tf.constant_initializer(0.0))
                conv3 = (tf.nn.conv2d(tf.concat([input,conv1,conv2],-1), weights, strides=[1, 1, 1, 1], padding='SAME') + bias)
                                                    
                conv3= tf.tanh(conv3)

        return conv3
        
    def PanSharpening_model(self,pan_img, ms_img):
        with tf.variable_scope('Pan_model'):
            if self.is_training:
                with tf.name_scope('upscale'):
                    # de_weight=tf.get_variable('de_weight', [3,3,self.num_spectrum, self.num_spectrum],
                                            # initializer=tf.truncated_normal_initializer(stddev=1e-3) )
                    # ms_scale4 = tf.nn.conv2d_transpose(ms_img, de_weight, output_shape=[self.batch_size,self.pan_size,self.pan_size,self.num_spectrum],
                                                       # strides=[1,4,4,1],padding='SAME' )                            
                    ms_img=tf.image.resize_images(ms_img, [self.pan_size, self.pan_size], method=2)
            input=tf.concat([ms_img,pan_img],axis=-1)
            with tf.variable_scope('layer1'):
                weights = tf.get_variable("w1", [9, 9, self.num_spectrum + 1, 64],
                                          initializer=tf.truncated_normal_initializer(stddev=1e-3))
                bias = tf.get_variable("b1", [64], initializer=tf.constant_initializer(0.0))
                conv1 = tf.contrib.layers.batch_norm(tf.nn.conv2d(input, weights, strides=[1, 1, 1, 1], padding='SAME') + bias,
                                                     decay=0.9,updates_collections=None, epsilon=1e-5, scale=True)
                conv1= tf.nn.relu(conv1)
            with tf.variable_scope('layer2'):
                weights = tf.get_variable("w2", [5, 5, 64, 32],
                                          initializer=tf.truncated_normal_initializer(stddev=1e-3))
                bias = tf.get_variable("b1", [32], initializer=tf.constant_initializer(0.0))
                conv2 = tf.contrib.layers.batch_norm(tf.nn.conv2d(conv1, weights, strides=[1, 1, 1, 1], padding='SAME') + bias,
                                                     decay=0.9,updates_collections=None, epsilon=1e-5, scale=True)
                conv2= tf.nn.relu(conv2)
            with tf.variable_scope('layer3'):
                weights = tf.get_variable("w3", [5, 5, 32 , self.num_spectrum],
                                          initializer=tf.truncated_normal_initializer(stddev=1e-3))
                bias = tf.get_variable("b3", [self.num_spectrum], initializer=tf.constant_initializer(0.0))
                conv3 = (tf.nn.conv2d(conv2, weights, strides=[1, 1, 1, 1], padding='SAME') + bias)
                                                    
                conv3= tf.tanh(conv3)

        return conv3
        
    def resnet_PanSharpening_model(self, pan_img, ms_img):
        def residual_block(inputs, output_channel, stride, scope):
            with tf.variable_scope(scope):
                net = Op.conv2(inputs, 3, output_channel, stride, use_bias=False, scope='conv_1')
                net = Op.batchnorm(net)
                net = Op.prelu_tf(net)
                net = Op.conv2(net, 3, output_channel, stride, use_bias=False, scope='conv_2')
                net = Op.batchnorm(net)
                net = net + inputs

            return net
            
        with tf.variable_scope('Pan_model'):
            if self.is_training:
                with tf.name_scope('upscale'):
                    ms_img=tf.image.resize_images(ms_img, [self.pan_size, self.pan_size], method=2)
            inputs=tf.concat([ms_img,pan_img],axis=-1)
            with tf.variable_scope('generator_unit'):
                # The input layer
                with tf.variable_scope('input_stage'):
                    net = Op.conv2(inputs, 9, 64, 1, scope='conv')
                    net = Op.prelu_tf(net)

                stage1_output = net

                # The residual block parts
                for i in range(1, 5+1 , 1):
                    name_scope = 'resblock_%d'%(i)
                    net = residual_block(net, 64, 1, name_scope)

                with tf.variable_scope('resblock_output'):
                    net = Op.conv2(net, 3, 4, 1, use_bias=False, scope='conv')
                    net= tf.tanh(net)
        
        return net
        
    def resnet_PanSharpening_model_dense(self, pan_img, ms_img):
        def residual_block(inputs, output_channel, stride, scope):
            with tf.variable_scope(scope):
                net1 = Op.conv2(inputs, 3, output_channel, stride, use_bias=False, scope='conv_1')
                net1 = Op.batchnorm(net1)
                net1 = tf.nn.relu(net1)
                net2 = Op.conv2(tf.concat([inputs,net1],-1), 3, output_channel, stride, use_bias=False, scope='conv_2')
                net2 = Op.batchnorm(net2)
                net2 = tf.nn.relu(net2)

            return net2
            
        with tf.variable_scope('Pan_model'):
            if self.is_training:
                with tf.name_scope('upscale'):
                    ms_img=tf.image.resize_images(ms_img, [self.pan_size, self.pan_size], method=2)
            inputs=tf.concat([ms_img,pan_img],axis=-1)
            with tf.variable_scope('generator_unit'):
                # The input layer
                with tf.variable_scope('input_stage'):
                    net = Op.conv2(inputs, 3, 64, 1, scope='conv')
                    net = Op.batchnorm(net)
                    net = tf.nn.relu(net)
                    net=tf.concat([inputs,net],-1)

                # The residual block parts
                for i in range(1, 5+1 , 1):
                    name_scope = 'resblock_%d'%(i)
                    net = residual_block(net, 64, 1, name_scope)

                with tf.variable_scope('resblock_output'):
                    net = Op.conv2(net, 3, 4, 1, use_bias=False, scope='conv')
                    net= tf.tanh(net)
        
        return net

    def spatial_discriminator(self,img_hp,reuse=False):
        with tf.variable_scope('spatial_discriminator', reuse=reuse):
            with tf.variable_scope('layer_1'):
                weights = tf.get_variable("w_1", [3, 3, 1, 16],
                                          initializer=tf.truncated_normal_initializer(stddev=1e-3))
                bias = tf.get_variable("b_1", [16], initializer=tf.constant_initializer(0.0))
                conv1_spatial = tf.nn.conv2d(img_hp, weights, strides=[1, 2, 2, 1], padding='SAME') + bias
                conv1_spatial = self.lrelu(conv1_spatial)
                # print(conv1_vi.shape)
            with tf.variable_scope('layer_2'):
                weights = tf.get_variable("w_2", [3, 3, 16, 32],
                                          initializer=tf.truncated_normal_initializer(stddev=1e-3))
                bias = tf.get_variable("b_2", [32], initializer=tf.constant_initializer(0.0))
                conv2_spatial = tf.contrib.layers.batch_norm(
                    tf.nn.conv2d(conv1_spatial, weights, strides=[1, 2, 2, 1], padding='SAME') + bias, decay=0.9,
                    updates_collections=None, epsilon=1e-5, scale=True)
                conv2_spatial = self.lrelu(conv2_spatial)
                # print(conv2_vi.shape)
            with tf.variable_scope('layer_3'):
                weights = tf.get_variable("w_3", [3, 3, 32, 64],
                                          initializer=tf.truncated_normal_initializer(stddev=1e-3))
                bias = tf.get_variable("b_3", [64], initializer=tf.constant_initializer(0.0))
                conv3_spatial = tf.contrib.layers.batch_norm(
                    tf.nn.conv2d(conv2_spatial, weights, strides=[1, 2, 2, 1], padding='SAME') + bias, decay=0.9,
                    updates_collections=None, epsilon=1e-5, scale=True)
                conv3_spatial = self.lrelu(conv3_spatial)
                # print(conv3_vi.shape)
            with tf.variable_scope('layer_4'):
                weights = tf.get_variable("w_4", [3, 3, 64, 128],
                                          initializer=tf.truncated_normal_initializer(stddev=1e-3))
                bias = tf.get_variable("b_4", [128], initializer=tf.constant_initializer(0.0))
                conv4_spatial = tf.contrib.layers.batch_norm(
                    tf.nn.conv2d(conv3_spatial, weights, strides=[1, 2, 2, 1], padding='SAME') + bias, decay=0.9,
                    updates_collections=None, epsilon=1e-5, scale=True)
                conv4_spatial = self.lrelu(conv4_spatial)
                #conv4_spatial = tf.reshape(conv4_spatial, [self.batch_size, 4 * 4 * 128])
            with tf.variable_scope('layer_5'):
                weights = tf.get_variable("w_5", [3, 3, 128, 256],
                                          initializer=tf.truncated_normal_initializer(stddev=1e-3))
                bias = tf.get_variable("b_5", [256], initializer=tf.constant_initializer(0.0))
                conv5_spatial = tf.contrib.layers.batch_norm(
                    tf.nn.conv2d(conv4_spatial, weights, strides=[1, 2, 2, 1], padding='SAME') + bias, decay=0.9,
                    updates_collections=None, epsilon=1e-5, scale=True)
                conv5_spatial = self.lrelu(conv5_spatial)
            with tf.variable_scope('line_6'):
                weights = tf.get_variable("w_6", [4, 4, 256, 1],
                                          initializer=tf.truncated_normal_initializer(stddev=1e-3))
                bias = tf.get_variable("b_6", [1], initializer=tf.constant_initializer(0.0))
                conv6_spatial=tf.contrib.layers.batch_norm(
                    tf.nn.conv2d(conv5_spatial, weights, strides=[1, 1, 1, 1], padding='VALID') + bias, decay=0.9,
                    updates_collections=None, epsilon=1e-5, scale=True)
                conv6_spatial = self.lrelu(conv6_spatial)
                conv6_spatial=tf.reshape(conv6_spatial, [self.batch_size, 1])
                #line5_spatial = tf.matmul(conv4_spatial, weights) + bias
                # conv3_vi= tf.contrib.layers.batch_norm(conv3_vi, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True)
        return conv6_spatial

    def spectrum_discriminator(self,img,reuse=False):
        with tf.variable_scope('spectrum_discriminator', reuse=reuse):
            with tf.variable_scope('layer1_spectrum'):
                weights = tf.get_variable("w1_spectrum", [3, 3, self.num_spectrum, 16],
                                          initializer=tf.truncated_normal_initializer(stddev=1e-3))
                bias = tf.get_variable("b1_spectrum", [16], initializer=tf.constant_initializer(0.0))
                conv1_spectrum = tf.nn.conv2d(img, weights, strides=[1, 2, 2, 1], padding='SAME') + bias
                conv1_spectrum = self.lrelu(conv1_spectrum)
                # print(conv1_vi.shape)
            with tf.variable_scope('layer2_spectrum'):
                weights = tf.get_variable("w2_spectrum", [3, 3, 16, 32],
                                          initializer=tf.truncated_normal_initializer(stddev=1e-3))
                bias = tf.get_variable("b2_spectrum", [32], initializer=tf.constant_initializer(0.0))
                conv2_spectrum = tf.contrib.layers.batch_norm(
                    tf.nn.conv2d(conv1_spectrum, weights, strides=[1, 2, 2, 1], padding='SAME') + bias, decay=0.9,
                    updates_collections=None, epsilon=1e-5, scale=True)
                conv2_spectrum = self.lrelu(conv2_spectrum)
                # print(conv2_vi.shape)
            with tf.variable_scope('layer3_spectrum'):
                weights = tf.get_variable("w3_spectrum", [3, 3, 32, 64],
                                          initializer=tf.truncated_normal_initializer(stddev=1e-3))
                bias = tf.get_variable("b3_spectrum", [64], initializer=tf.constant_initializer(0.0))
                conv3_spectrum = tf.contrib.layers.batch_norm(
                    tf.nn.conv2d(conv2_spectrum, weights, strides=[1, 2, 2, 1], padding='SAME') + bias, decay=0.9,
                    updates_collections=None, epsilon=1e-5, scale=True)
                conv3_spectrum = self.lrelu(conv3_spectrum)
                # print(conv3_vi.shape)
            with tf.variable_scope('layer4_spectrum'):
                weights = tf.get_variable("w4_spectrum", [3, 3, 64, 128],
                                          initializer=tf.truncated_normal_initializer(stddev=1e-3))
                bias = tf.get_variable("b4_spectrum", [128], initializer=tf.constant_initializer(0.0))
                conv4_spectrum = tf.contrib.layers.batch_norm(
                    tf.nn.conv2d(conv3_spectrum, weights, strides=[1, 2, 2, 1], padding='SAME') + bias, decay=0.9,
                    updates_collections=None, epsilon=1e-5, scale=True)
                conv4_spectrum = self.lrelu(conv4_spectrum)
                #conv4_spectrum = tf.reshape(conv4_spectrum, [self.batch_size, 1 * 1 * 128])
            with tf.variable_scope('layer5_spectrum'):
                weights = tf.get_variable("w5_spectrum", [3, 3, 128, 256],
                                          initializer=tf.truncated_normal_initializer(stddev=1e-3))
                bias = tf.get_variable("b5_spectrum", [256], initializer=tf.constant_initializer(0.0))
                conv5_spectrum = tf.contrib.layers.batch_norm(
                    tf.nn.conv2d(conv4_spectrum, weights, strides=[1, 2, 2, 1], padding='SAME') + bias, decay=0.9,
                    updates_collections=None, epsilon=1e-5, scale=True)
                conv5_spectrum = self.lrelu(conv5_spectrum)
                #conv4_spectrum = tf.reshape(conv4_spectrum, [self.batch_size, 1 * 1 * 128])
            with tf.variable_scope('line6_spectrum'):
                weights = tf.get_variable("w6_spectrum", [4 , 4 , 256, 1],
                                          initializer=tf.truncated_normal_initializer(stddev=1e-3))
                bias = tf.get_variable("b6_spectrum", [1], initializer=tf.constant_initializer(0.0))
                conv6_spectrum=tf.contrib.layers.batch_norm(
                    tf.nn.conv2d(conv5_spectrum, weights, strides=[1, 1, 1, 1], padding='VALID') + bias, decay=0.9,
                    updates_collections=None, epsilon=1e-5, scale=True)
                conv6_spectrum=self.lrelu(conv6_spectrum)
                conv6_spectrum=tf.reshape(conv6_spectrum, [self.batch_size, 1])
                #line5_spectrum = tf.matmul(conv4_spectrum, weights) + bias
                # conv3_vi= tf.contrib.layers.batch_norm(conv3_vi, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True)
        return conv6_spectrum
        
    def high_pass(self, img, type='PanSharepening'):
        if type=='pan':
            input=img
            for i in range(self.num_spectrum-1):
                input=tf.concat([input,img],axis=-1)
            img=input
        blur_kerel=np.zeros(shape=(13,13,self.num_spectrum, self.num_spectrum), dtype=np.float32)
        value=1/169*np.ones(shape=(13,13), dtype=np.float32)
        for i in range(self.num_spectrum):
            blur_kerel[:,:,i,i]=value
        img_lp=tf.nn.conv2d(img,tf.convert_to_tensor(blur_kerel),strides=[1,1,1,1], padding='SAME')
        img_hp=tf.reshape(tf.reduce_mean(img-img_lp,3),[self.batch_size,self.pan_size,self.pan_size,1])
        return tf.abs(img_hp)
        
    def blur(self, img, kernel_size,gaussian_variance=1):
        blur_kerel=np.zeros(shape=(kernel_size, kernel_size, self.num_spectrum, self.num_spectrum), dtype=np.float32)
        value=self.generate_Gauss(kernel_size)
        for i in range(self.num_spectrum):
            blur_kerel[:,:,i,i]=value
        img_blur=tf.nn.conv2d(img,tf.convert_to_tensor(blur_kerel),strides=[1,1,1,1], padding='SAME')
        return img_blur
        
    def high_pass_1(self, img, type='PanSharepening'):
        if type=='pan':
            input=img
            for i in range(3):
                input=tf.concat([input,img],axis=-1)
            img=input
        blur_kerel=np.zeros(shape=(3,3,4,4), dtype=np.float32)
        value=np.array([[1.,1.,1.],[1.,-8.,1.],[1.,1.,1.]])
        for i in range(4):
            blur_kerel[:,:,i,i]=value
        img_hp=tf.nn.conv2d(img,tf.convert_to_tensor(blur_kerel),strides=[1,1,1,1], padding='SAME')
        img_hp=tf.reshape(tf.reduce_mean(img_hp,3),[self.batch_size,128,128,1])
        #img_hp=img-img_lp
        return tf.abs(img_hp)

    ####2019-11-17
    def high_pass_2(self, img, type='PanSharepening'):

        blur_kerel=np.zeros(shape=(3,3,1,1), dtype=np.float32)
        value=np.array([[1.,1.,1.],[1.,-8.,1.],[1.,1.,1.]])
        blur_kerel[:,:,0,0]=value
        img_hp=tf.nn.conv2d(img,tf.convert_to_tensor(blur_kerel),strides=[1,1,1,1], padding='SAME')
        #img_hp=tf.reshape(tf.reduce_mean(img_hp,3),[self.batch_size,128,128,1])
        #img_hp=img-img_lp
        return img_hp
        
    def generate_Gauss(self,kernel_size,gaussian_variance=1):
        x = np.arange(0, kernel_size, 1, float)
        y = x[:, np.newaxis]
        x0 = y0 = kernel_size // 2
        kernel=np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / 2.0 / gaussian_variance/ gaussian_variance)
        kernel=kernel/np.sum(kernel)
        return kernel
       
    def lrelu(self,x, leak=0.2):
        return tf.maximum(x, leak * x)

    def AddChannel(self, x):
        input=x
        for i in range(self.num_spectrum-1):
            input=tf.concat([input,x],axis=-1)
        return input
