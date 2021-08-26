#%% imports
"""
Created on Sun Dec  6 19:30:24 2020

@author: Raz
"""
import cv2
import matplotlib.pyplot as plt
import numpy as np
import h5py
from sklearn.utils import shuffle
import tensorflow as tf
import os
from sklearn import metrics
from tqdm import tqdm
#%% Constants
##############################################################################
train_files = ['SynthText.h5',
              'SweetPuppy.h5',
              'UbuntuMono.h5',
              'Skylark.h5']

validation_file = 'SynthText_val.h5'

font_names = ["Skylark", "Sweet Puppy", "Ubuntu Mono"]
paths = ['training_data','validation_data']
img_size = 100
epochs = 100
filepath = "cnn-mynetwork.hdf5"
##############################################################################
#%% preprocessing the train images, and saving new images into 

data_imgs = [[],[],[]]
data_lbls = [[],[],[]]
index_letter = 0
for filename in tqdm(train_files,desc="preproccesing the train files",colour="blue"):
    db = h5py.File(filename)
    im_names = list(db['data'].keys())    
    for im_name in im_names:
        img = db['data'][im_name][:]
        font = db['data'][im_name].attrs['font']
        txt = db['data'][im_name].attrs['txt']
        charBB = db['data'][im_name].attrs['charBB']
        wordBB = db['data'][im_name].attrs['wordBB']
        i = 0
        for word in txt:
            for char in word:
                if(char == ' ' or char == '\n'):
                    continue
                original = np.float32([charBB[:, :, i].T[0], charBB[:, :, i].T[1], charBB[:, :, i].T[3], charBB[:, :, i].T[2]])
                new_pic = np.float32([[0, 0], [img_size, 0], [0, img_size], [img_size, img_size]])
                transform = cv2.getPerspectiveTransform(original, new_pic)
                letter = cv2.warpPerspective(img, transform, (img_size, img_size)) 
                curr_font = font[i]
                i += 1
                data_imgs[font_names.index(curr_font.decode('UTF-8'))].append(letter)
                name = (curr_font.decode('UTF-8')).replace(" ", "") +"_"+str(index_letter)+".jpg"
                data_lbls[font_names.index(curr_font.decode('UTF-8'))].append(name)
                index_letter +=1

# shuffle all the images, creating balanced dataset-
# taking only the minimum from all of the labels
min_imgs = len(data_imgs[0])
for i in range(len(font_names)):
    data_imgs[i],data_lbls[i] = shuffle(data_imgs[i],data_lbls[i])
    min_imgs = np.minimum(len(data_imgs[i]), min_imgs)

for i in range(len(font_names)):
    data_imgs[i] = data_imgs[i][:min_imgs]

# creating files for the training and the validation pictures
for path in paths:
    if not os.path.isdir(path):
        os.mkdir(path)
        for name in font_names:
            os.mkdir(os.path.join(path, name.replace(" ", "")))
            
# saving after spliting to validation and training
training_sz = int(0.8*min_imgs)
path_index = 0
train_sz_finished = False
for i in tqdm(range(min_imgs),desc="saving new images",colour="yellow"):
    if(i > training_sz and train_sz_finished == False):
        path_index +=1
        train_sz_finished = True
    for j in range(len(font_names)):
        cv2.imwrite(os.path.join(os.path.join(paths[path_index], font_names[j].replace(" ", ""),data_lbls[j][i])), data_imgs[j][i])

#%% get data into keras using image_dataset_from_directory
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    paths[0],
    labels="inferred",
    label_mode="categorical",
    class_names=None,
    color_mode="grayscale",
    batch_size=32,
    image_size=(img_size, img_size),
    shuffle=True,
    seed=1,
    validation_split=0,
    interpolation="bilinear",
    follow_links=False,
)
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    paths[1],
    labels="inferred",
    label_mode="categorical",
    class_names=None,
    color_mode="grayscale",
    batch_size=32,
    image_size=(img_size, img_size),
    shuffle=True,
    seed=1,
    validation_split=0,
    interpolation="bilinear",
    follow_links=False,
)


#%% data augmentation of the model
# custom layer - adding salt and pepper layer
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

# https://towardsdatascience.com/how-to-reduce-training-time-for-a-deep-learning-model-using-tf-data-43e1989d2961
data_augmentation = tf.keras.models.Sequential(
    [
     tf.keras.layers.experimental.preprocessing.RandomContrast(0.3),
     tf.keras.layers.experimental.preprocessing.RandomZoom(0.1),
     tf.keras.layers.GaussianNoise(0.1),
     SaltAndPepper(0.1),
    ]
)



#%% https://www.kaggle.com/sanikamal/font-classification-using-cnn

# the model
model = tf.keras.models.Sequential([
    data_augmentation,
    # normalize the picture after the data augmentation
    tf.keras.layers.experimental.preprocessing.Rescaling(1. / 255),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(256, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(256, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Conv2D(512, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
])

# model compilation features
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1e-2,
        decay_steps=10000,
        decay_rate=0.9)

optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule)

model.compile(loss='categorical_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])


callbacks_list = [
    tf.keras.callbacks.ModelCheckpoint(filepath,
                                       monitor='val_accuracy',
                                       verbose=1,
                                       save_best_only=True,
                                       mode='max')
    ]

# fitting the model, saving into histogram in order to show the results
histgram = model.fit(train_ds,
                    batch_size=32,
                    epochs=epochs,
                    validation_data=val_ds,
                    callbacks=callbacks_list)
# printing model summary
model.summary()

# showing the results
plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(range(epochs),  histgram.history['accuracy'], label='Training Accuracy')
plt.plot(range(epochs), histgram.history['val_accuracy'], label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')
plt.xlabel("epochs")
plt.ylabel("percentage")

plt.subplot(1, 2, 2)
plt.plot(range(epochs), histgram.history['loss'], label='Training Loss')
plt.plot(range(epochs), histgram.history['val_loss'], label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.xlabel("epochs")
plt.ylabel("percentage")
plt.show()

#%% testing on validation file, showing classification report

db = h5py.File(validation_file)
im_names = list(db['data'].keys())    
letters_pred=[[],[]]
words_pred=[[],[]]
for im_name in tqdm(im_names,desc="preproccesing and computing the validation images",colour="red"):
    img=db['data'][im_name][:]
    charBB = db['data'][im_name].attrs['charBB']
    txt = db['data'][im_name].attrs['txt']
    font = db['data'][im_name].attrs['font']
    i = 0
    for words in txt:
        word=[]
        font_name=font[i].decode('UTF-8')
        for char in words:
            original = np.float32([charBB[:, :, i].T[0], charBB[:, :, i].T[1], charBB[:, :, i].T[3], charBB[:, :, i].T[2]])
            new_pic = np.float32([[0, 0], [img_size, 0], [0, img_size], [img_size, img_size]])
            transform = cv2.getPerspectiveTransform(original, new_pic)
            letter = cv2.warpPerspective(img, transform, (img_size, img_size)) 
            letter= cv2.cvtColor(letter,cv2.COLOR_BGR2GRAY)
            word.append(letter)
            i += 1
        
        # computing for the whole word vs each letter
        font_prob=[0,0,0]
        # summing for all the letters, choosing the max of all
        for letter in word:
            img_ready = tf.expand_dims(tf.keras.preprocessing.image.img_to_array(letter), 0)
            font_prob += model.predict(img_ready)[0]    
            
        predictedFont=font_names[np.argmax(font_prob)] 
        # 
        for letter in word:
            letters_pred[0].append(font_name)
            letters_pred[1].append(predictedFont)
        words_pred[0].append(font_name)
        words_pred[1].append(predictedFont)

print("Word score")
print(metrics.classification_report(*letters_pred))
print("Letter score")
print(metrics.classification_report(*words_pred))