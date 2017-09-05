# -*- coding: utf-8 -*-

import model
import util
import tensorflow as tf
import numpy as np


train_dir='/home/yejg/tensorflow/eye/data/train/'
batch_size=32
img_w=256
img_h=256
d_lr=0.0002
g_lr=0.0001
z_dim=100
d_depths=[128,256,512]
g_depths=[512,256,128]
s_size=32     #通过改变s_size大小来改变generated的大小，使得跟照片大小一致
z_dim=256
MAX_STEP=10

x=tf.placeholder(tf.float32,shape=[batch_size,img_w,img_h,3])
z=tf.placeholder(tf.float32,shape=[batch_size,z_dim])
#g=model.Generator(depths=[512,256,128],s_size=s_size)
#gen=g(z,training=True)
#
#d=model.Discriminator(depths=[64,128,256])
#dl=d(x,training=True)
#
#
#wgan=model.WGAN(batch_size,d_depths=d_depths,s_size=s_size)
#
#losses=wgan.loss(x)
#
#train_op=wgan.train(losses)

#g=model.Generator(s_size=s_size)
#generated=g.__call__(z,training=True)  #shape=(64, 512, 512, 3)

###  get train_op and loss
wgan=model.WGAN(batch_size,d_depths=d_depths,g_depths=g_depths,s_size=s_size)
losses=wgan.loss(x)     #果然是要generated的图片与真是的对应问题
train_op=wgan.train(losses=losses,d_lr=d_lr,g_lr=g_lr)

d_clip_variable=[p.assign(tf.clip_by_value(p,-1.0,1.0)) for p in wgan.d.variables]


crop=util.resize(res_img_h=img_h,res_img_w=img_w)
image_files,image_labels=util.get_files(train_dir)


#training
sess=tf.Session()
sess.run(tf.global_variables_initializer())

#image_batch=np.array(
#            map(crop.crop_resize,image_files[:32])).astype(np.float32)
#
#_, g_loss_value, d_loss_value = sess.run([train_op, losses[wgan.g], losses[wgan.d]],
#                                               feed_dict={x:image_batch})
#
#print "g_loss: ------ %f\td_loss: -------%f"%(g_loss_value,d_loss_value)



iteration=0
k=2
for step in range(MAX_STEP):
   
   for start,end in zip(range(0,len(image_files),batch_size),
                        range(batch_size,len(image_files),batch_size)):
      image_batch_list=image_files[start:end]
      image_batch=np.array(
            map(crop.crop_resize,image_batch_list)).astype(np.float32)
      
      _, g_loss_value, d_loss_value = sess.run([train_op, losses[wgan.g], losses[wgan.d]],
                                               feed_dict={x:image_batch})
      
      if iteration%k==0:
            print "g_loss: ------ %f\td_loss: -------%f"%(g_loss_value,d_loss_value)
      
      iteration+=2
      
      













