# -*- coding: utf-8 -*-
#%%
import tensorflow as tf
import util
from glob import glob
import os
import numpy as np
#%%

class Generator:
    def __init__(self, depths=[1024, 512, 256, 128], s_size=4):
        self.depths = depths + [3]
        self.s_size = s_size
        self.reuse = False

    def __call__(self, inputs, training=False):
        inputs = tf.convert_to_tensor(inputs)
        with tf.variable_scope('g', reuse=self.reuse):
            # reshape from inputs
            with tf.variable_scope('reshape'):
                outputs = tf.layers.dense(inputs, self.depths[0] * self.s_size * self.s_size)
                outputs = tf.reshape(outputs, [-1, self.s_size, self.s_size, self.depths[0]])
                outputs = tf.nn.relu(tf.layers.batch_normalization(outputs, training=training), name='outputs')
            # deconvolution (transpose of convolution) x 4
#            with tf.variable_scope('deconv1'):
#                outputs = tf.layers.conv2d_transpose(outputs, self.depths[1], [5, 5], strides=(2, 2), padding='SAME')
#                outputs = tf.nn.relu(tf.layers.batch_normalization(outputs, training=training), name='outputs')
#            with tf.variable_scope('deconv2'):
#                outputs = tf.layers.conv2d_transpose(outputs, self.depths[2], [5, 5], strides=(2, 2), padding='SAME')
#                outputs = tf.nn.relu(tf.layers.batch_normalization(outputs, training=training), name='outputs')
#            with tf.variable_scope('deconv3'):
#                outputs = tf.layers.conv2d_transpose(outputs, self.depths[3], [5, 5], strides=(2, 2), padding='SAME')
#                outputs = tf.nn.relu(tf.layers.batch_normalization(outputs, training=training), name='outputs')
#            with tf.variable_scope('deconv4'):
#                outputs = tf.layers.conv2d_transpose(outputs, self.depths[4], [5, 5], strides=(2, 2), padding='SAME')
            # output images
            
            n=len(self.depths)
            for i in range(n-1):
               with tf.variable_scope('deconv_%d'%(i+1)):
                  outputs=tf.layers.conv2d_transpose(outputs,self.depths[i+1],[5,5],strides=(2,2),padding='SAME')
                  
                  if (i+1)<(n-1):
                     outputs = tf.nn.relu(tf.layers.batch_normalization(outputs, training=training), name='outputs')
                  else:
                     outputs=outputs

            with tf.variable_scope('tanh'):
                outputs = tf.tanh(outputs, name='outputs')
        self.reuse = True
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='g')
        return outputs


class Discriminator:
    def __init__(self, depths=[64, 128, 256, 512]):
        self.depths = [3] + depths
        self.reuse = False

    def __call__(self, inputs, training=False, name=''):
        def leaky_relu(x, leak=0.2, name=''):
            return tf.maximum(x, x * leak, name=name)
        outputs = tf.convert_to_tensor(inputs)

        with tf.name_scope('d' + name), tf.variable_scope('d', reuse=self.reuse):
            # convolution x 4
#            with tf.variable_scope('conv1'):
#                outputs = tf.layers.conv2d(outputs, self.depths[1], [5, 5], strides=(2, 2), padding='SAME')
#                outputs = leaky_relu(tf.layers.batch_normalization(outputs, training=training), name='outputs')
#            with tf.variable_scope('conv2'):
#                outputs = tf.layers.conv2d(outputs, self.depths[2], [5, 5], strides=(2, 2), padding='SAME')
#                outputs = leaky_relu(tf.layers.batch_normalization(outputs, training=training), name='outputs')
#            with tf.variable_scope('conv3'):
#                outputs = tf.layers.conv2d(outputs, self.depths[3], [5, 5], strides=(2, 2), padding='SAME')
#                outputs = leaky_relu(tf.layers.batch_normalization(outputs, training=training), name='outputs')
#            with tf.variable_scope('conv4'):
#                outputs = tf.layers.conv2d(outputs, self.depths[4], [5, 5], strides=(2, 2), padding='SAME')
#                outputs = leaky_relu(tf.layers.batch_normalization(outputs, training=training), name='outputs')
#            with tf.variable_scope('classify'):
#                batch_size = outputs.get_shape()[0].value
#                reshape = tf.reshape(outputs, [batch_size, -1])
#                outputs = tf.layers.dense(reshape, 2, name='outputs')
            
             n=len(self.depths)
             for i in range(n-1):
                with tf.variable_scope('conv_%d'%(i+1)):
                   outputs=tf.layers.conv2d(outputs,self.depths[i+1],[5,5],strides=(2,2),padding='SAME')
                   outputs=leaky_relu(tf.layers.batch_normalization(outputs,training=training),name='output')
                   
             with tf.variable_scope('classify'):
                batch_size = outputs.get_shape()[0].value
                reshape = tf.reshape(outputs, [batch_size, -1])
                outputs = tf.layers.dense(reshape, 2, name='outputs')
                     
        self.reuse = True
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='d')
        return outputs


class WGAN:
    def __init__(self,
                 batch_size=128, s_size=4, z_dim=100,
                 g_depths=[1024, 512, 256, 128],
                 d_depths=[64, 128, 256, 512]):
        self.batch_size = batch_size
        self.s_size = s_size
        self.z_dim = z_dim
        self.g = Generator(depths=g_depths, s_size=self.s_size)
        self.d = Discriminator(depths=d_depths)
        self.z = tf.random_uniform([self.batch_size, self.z_dim], minval=-1.0, maxval=1.0)

    def loss(self, traindata):
        """build models, calculate losses.
        Args:
            traindata: 4-D Tensor of shape `[batch, height, width, channels]`.
        Returns:
            dict of each models' losses.
        """
        generated = self.g(self.z, training=True)
        t_outputs = self.d(traindata, training=True, name='t')
#        self.d.reuse=True
        g_outputs = self.d(generated, training=True, name='g')
        # add each losses to collection
#        tf.add_to_collection(
#            'g_losses',
#            tf.reduce_mean(
#                tf.nn.sparse_softmax_cross_entropy_with_logits(
#                    labels=tf.ones([self.batch_size], dtype=tf.int64),
#                    logits=g_outputs)))
#        tf.add_to_collection(
#            'd_losses',
#            tf.reduce_mean(
#                tf.nn.sparse_softmax_cross_entropy_with_logits(
#                    labels=tf.ones([self.batch_size], dtype=tf.int64),
#                    logits=t_outputs)))
#        
#        tf.add_to_collection(
#            'd_losses',
#            tf.reduce_mean(
#                tf.nn.sparse_softmax_cross_entropy_with_logits(
#                    labels=tf.zeros([self.batch_size], dtype=tf.int64),
#                    logits=g_outputs)))
#        return {
#            self.g: tf.add_n(tf.get_collection('g_losses'), name='total_g_loss'),
#            self.d: tf.add_n(tf.get_collection('d_losses'), name='total_d_loss'),
#        }
        d_real=tf.reduce_mean(
              tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.ones([self.batch_size],dtype=tf.int64),
                                                             logits=t_outputs))
        d_fake=tf.reduce_mean(
              tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.zeros([self.batch_size],dtype=tf.int64),
                                                             logits=g_outputs))
        d_loss=d_real+d_fake
        g_loss=-tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=tf.ones([self.batch_size], dtype=tf.int64),logits=g_outputs))
        
        return  {
            self.g:g_loss,
            self.d:d_loss,
        }
        
    def train(self, losses, d_lr=0.0002,g_lr=0.0001):
        """
        Args:
            losses dict.
        Returns:
            train op.
        """
        
        g_opt = tf.train.RMSPropOptimizer(learning_rate=g_lr)
        d_opt = tf.train.RMSPropOptimizer(learning_rate=d_lr)
        g_opt_op = g_opt.minimize(losses[self.g], var_list=self.g.variables)
        d_opt_op = d_opt.minimize(-losses[self.d], var_list=self.d.variables)
        with tf.control_dependencies([g_opt_op, d_opt_op]):
            return tf.no_op(name='train')

    def sample_images(self, row=8, col=8, inputs=None):
        if inputs is None:
            inputs = self.z
        images = self.g(inputs, training=True)
        images = tf.image.convert_image_dtype(tf.div(tf.add(images, 1.0), 2.0), tf.uint8)
        images = [image for image in tf.split(images, self.batch_size, axis=0)]
        rows = []
        for i in range(row):
            rows.append(tf.concat(images[col * i + 0:col * i + col], 2))
        image = tf.concat(rows, 1)
        return tf.image.decode_jpeg(tf.squeeze(image,[0]))
#%%