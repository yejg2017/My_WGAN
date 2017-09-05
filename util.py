# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import math
import cv2
 #%%
train_dir='/home/yejg/tensorflow/eye/data/train/'
 
def get_files(file_dir):
    '''
    Args:
        file_dir: file directory
    Returns:
        list of images and labels
    '''
    health = []
    label_health = []
    sick= []
    label_sick = []
    for file in os.listdir(file_dir):
        for f in os.listdir(os.path.join(file_dir,file)):
           if file=='health':
              health.append(file_dir+file+'/'+f)
              label_health.append(1)
              
           if file=='sick':
              sick.append(file_dir+file+'/'+f)
              label_sick.append(0)
           

#    print('There are %d cats\nThere are %d dogs' %(len(cats), len(dogs)))
    
    image_list = np.hstack((health,sick))
    label_list = np.hstack((label_health, label_sick))
    
    temp = np.array([image_list, label_list])
    temp = temp.transpose()
    np.random.shuffle(temp)   
    
    all_image_list = temp[:, 0]
    all_label_list = temp[:, 1]
    
    all_label_list=[int(float(i)) for i in all_label_list]
    
    return all_image_list,all_label_list
 
#images,labels=get_files(train_dir)
#%%

def get_batch(image, label, image_W, image_H, batch_size, capacity):
    '''
    Args:
        image: list type
        label: list type
        image_W: image width
        image_H: image height
        batch_size: batch size
        capacity: the maximum elements in queue
    Returns:
        image_batch: 4D tensor [batch_size, width, height, 3], dtype=tf.float32
        label_batch: 1D tensor [batch_size], dtype=tf.int32
    '''
    
    image = tf.cast(image, tf.string)
    label = tf.cast(label, tf.int32)

    # make an input queue
    input_queue = tf.train.slice_input_producer([image, label])
    
    label = input_queue[1]
    image_contents = tf.read_file(input_queue[0])
    image = tf.image.decode_jpeg(image_contents, channels=3)
    
    ######################################
    # data argumentation should go to here
    ######################################
    
    image = tf.image.resize_image_with_crop_or_pad(image, image_W, image_H)    
    # if you want to test the generated batches of images, you might want to comment the following line.
    
    #image = tf.image.per_image_standardization(image)
    
    image_batch, label_batch = tf.train.batch([image, label],
                                                batch_size= batch_size,
                                                num_threads= 64, 
                                                capacity = capacity)
    
    label_batch = tf.reshape(label_batch, [batch_size])
    image_batch = tf.cast(image_batch, tf.float32)
    
    return image_batch, label_batch
 
#%%
class resize:
   def __init__(self,
                res_img_h,
                res_img_w):
      #self.image_path=image_path
      self.res_img_h=res_img_h
      self.res_img_w=res_img_w
      
   def crop_resize(self,image_path):
    image = cv2.imread(image_path)
    height, width, channel = image.shape

    if width == height:
        resized_image = cv2.resize(image, (self.res_img_h,self.res_img_w))
    elif width > height:
        resized_image = cv2.resize(image, (int(width * float(self.res_img_h)/height), self.res_img_w))
        cropping_length = int( (resized_image.shape[1] - self.res_img_h) / 2)
        resized_image = resized_image[:,cropping_length:cropping_length+self.res_img_w]
    else:
        resized_image = cv2.resize(image, (self.res_img_h, int(height * float(self.res_img_w)/width)))
        cropping_length = int( (resized_image.shape[0] - self.res_img_w )/ 2)
        resized_image = resized_image[cropping_length:cropping_length+self.res_img_h, :]

    return (resized_image - 127.5) / 127.5
#%%
#images_list,label_list=get_files(train_dir)
#images,labels=get_batch(images_list,label_list,720,720,32,356)

#%%


