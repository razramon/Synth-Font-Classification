import cv2
import matplotlib.pyplot as plt
import numpy as np
import h5py
import random
import os


file_names = ['font_recognition_train_set/SynthText.h5',
              'SynthText/results/SweetPuppy.h5',
              'SynthText/results/UbuntuMono.h5',
              'SynthText/results/Skylark.h5']
font_names = ["Skylark", "Sweet Puppy", "Ubuntu Mono"]
cnt_fonts = [0,0,0]
cnt_fonts_words = [0,0,0]
for filename in file_names:
    db = h5py.File(filename)
    im_names = list(db['data'].keys())  
   
    for im_name in im_names:
        img = db['data'][im_name][:]
        font = db['data'][im_name].attrs['font']
        txt = db['data'][im_name].attrs['txt']

        charBB = db['data'][im_name].attrs['charBB']
        wordBB = db['data'][im_name].attrs['wordBB']
        i = 0
        for words in txt: # taking every word in the txt and parsing chars
            for char in words:
                if(char == ' ' or char == '\n'):
                    continue
                cnt_fonts[font_names.index(font[i].decode('UTF-8'))] += 1
            cnt_fonts_words[font_names.index(font[i].decode('UTF-8'))] += 1
            i+=1
min_occur = min(cnt_fonts)
y_pos = np.arange(len(font_names))
plt.figure(figsize=(15, 8))
plt.subplot(1, 3, 1)
plt.bar(y_pos,cnt_fonts, align='center', alpha=0.8,color=['red', 'green', 'blue'])
plt.xticks(y_pos, font_names)
plt.ylabel('Occurrences')
plt.title('Letters per Font')
for i in range(len(cnt_fonts)):
    plt.annotate(str(cnt_fonts[i]), xy=(i,cnt_fonts[i]), ha='center', va='bottom')
plt.subplot(1, 3, 2)
y_pos = np.arange(len(font_names))
plt.bar(y_pos,cnt_fonts_words, align='center', alpha=0.8,color=['red', 'green', 'blue'])
plt.xticks(y_pos, font_names)
plt.ylabel('Occurrences')
plt.title('Words per Font')
for i in range(len(cnt_fonts_words)):
    plt.annotate(str(cnt_fonts_words[i]), xy=(i,cnt_fonts_words[i]), ha='center', va='bottom')
plt.subplot(1,3, 3)
arr = [min_occur,min_occur,min_occur]
plt.bar(y_pos,arr, align='center', alpha=0.8,color=['red', 'green', 'blue'])
plt.xticks(y_pos, font_names)
plt.ylabel('Occurrences')
plt.title('Letters per Font Chosen')
for i in range(len(arr)):
    plt.annotate(str(arr[i]), xy=(i,arr[i]), ha='center', va='bottom')
plt.show()