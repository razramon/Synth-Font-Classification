# -*- coding: utf-8 -*-
"""
Created on Sun Jan 17 22:37:16 2021

@author: Raz
"""

import cv2
import numpy as np
import h5py
import tensorflow as tf
from tensorflow.keras.models import load_model
import csv
from tqdm import tqdm

# custom layer
class SaltAndPepper(tf.keras.layers.Layer):

    def __init__(self, ratio, **kwargs):
        super(SaltAndPepper, self).__init__(**kwargs)
        self.supports_masking = True
        self.ratio = ratio

    # the definition of the call method of custom layer
    def call(self, inputs, training=None):
        def noised():
            shp = tf.keras.backend.shape(inputs)[1:]
            mask_select = tf.keras.backend.random_binomial(shape=shp, p=self.ratio)
            mask_noise = tf.keras.backend.random_binomial(shape=shp, p=0.5) # salt and pepper have the same chance
            out = inputs * (1-mask_select) + mask_noise * mask_select
            return out

        return tf.keras.backend.in_train_phase(noised(), inputs, training=training)

    def get_config(self):
        config = {'ratio': self.ratio}
        base_config = super(SaltAndPepper, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

#Constants
##############################################################################

model = load_model('cnn-mynetwork.hdf5', custom_objects={'SaltAndPepper': SaltAndPepper})
file_name = 'test.h5'
img_sz, img_channel = 100, 1
##############################################################################

db = h5py.File(file_name)
im_names = list(db['data'].keys())    
font_names = ["Skylark", "Sweet Puppy", "Ubuntu Mono"]
index_letter = 0

with open('predictions.csv',mode='w',newline='') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow([' ', 'image', 'char', "b'Skylark'", "b'Sweet Puppy'", "b'Ubuntu Mono'"])
    for im_name in tqdm(im_names):
        img=db['data'][im_name][:]
        charBB = db['data'][im_name].attrs['charBB']
        txt = db['data'][im_name].attrs['txt']
        i = 0
        for words in txt:
            word=[]
            for char in words:  
                original = np.float32([charBB[:, :, i].T[0], charBB[:, :, i].T[1], charBB[:, :, i].T[3], charBB[:, :, i].T[2]])
                new_pic = np.float32([[0, 0], [img_sz, 0], [0, img_sz], [img_sz, img_sz]])
                transform = cv2.getPerspectiveTransform(original, new_pic)
                letter = cv2.warpPerspective(img, transform, (img_sz, img_sz)) 
                if(img_channel == 1):
                    letter = cv2.cvtColor(letter,cv2.COLOR_BGR2GRAY)
                word.append(letter)
                i += 1
                
            probability=[0,0,0] 
            for letter in word:
                img_array = tf.keras.preprocessing.image.img_to_array(letter)
                img_array = tf.expand_dims(img_array, 0)
                probability += model.predict(img_array)[0]
                
            predict_word=np.argmax(probability)
            font_predict = [0,0,0]
            font_predict[predict_word] = 1
            
            for letter_index in range(len(word)): 
                writer.writerow([index_letter, im_name, chr(words[letter_index]), *font_predict])
                index_letter +=1
