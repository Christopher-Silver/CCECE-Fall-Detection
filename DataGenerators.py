from google.colab import drive
drive.mount('/content/drive')
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_data_dir = '/content/drive/MyDrive/fall_detection/train'
val_data_dir = '/content/drive/MyDrive/fall_detection/validation'
from tensorflow.keras.preprocessing.image import ImageDataGenerator

batch_size = 1
target_size = (256, 256)
num_frames = 10


train_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=target_size,
        batch_size=batch_size,
        classes=['fall', 'non-fall'],
        color_mode = 'grayscale',
        shuffle = False,
        class_mode='binary')

val_datagen = ImageDataGenerator(rescale=1./255)
val_generator = val_datagen.flow_from_directory(
        val_data_dir,
        target_size=target_size,
        batch_size=batch_size,
        classes=['fall', 'non-fall'],
        color_mode = 'grayscale',
        shuffle = False,
        class_mode='binary')

import random
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
print(train_generator.filenames)


def train_data_generator(train_generator, batch_size):
    while True:
        X_batch = []
        y_batch = []
        for i in range(batch_size):
            # randomly select a video
            video_name = os.path.dirname(random.choice(train_generator.filepaths))
            frames_dir = os.path.join(train_generator.directory, video_name)
            frames = [f for f in os.listdir(frames_dir) if f.endswith('.jpg')]
            frames.sort()
            # load all frames from the video directory
            frames_array = []
            for j in range(len(frames)):
                img_path = os.path.join(frames_dir, frames[j])
                img = load_img(img_path, target_size=(256, 256), color_mode='grayscale')
                img_array = img_to_array(img)
                frames_array.append(img_array)
            frames_array = np.array(frames_array)
            X_batch.append(frames_array)
            # determine the label based on whether the video contains a fall or not
            if 'non-fall' in video_name:
              y_batch.append(0)
            else:
              y_batch.append(1)

        X_batch = np.array(X_batch)
        #print(X_batch)
        y_batch = np.array(y_batch)
        yield X_batch, y_batch

def val_data_generator(val_generator, batch_size):
    while True:
        XVal_batch = []
        yVal_batch = []
        for i in range(batch_size):
            # randomly select a video
            video_nameVal = os.path.dirname(random.choice(val_generator.filepaths))
            frames_dirVal = os.path.join(val_generator.directory, video_nameVal)
            framesVal = [f for f in os.listdir(frames_dirVal) if f.endswith('.jpg')]
            framesVal.sort()
            # load all frames from the video directory
            frames_arrayVal = []
            for j in range(len(framesVal)):
                img_pathVal = os.path.join(frames_dirVal, framesVal[j])
                imgVal = load_img(img_pathVal, target_size=(256, 256), color_mode='grayscale')
                img_arrayVal = img_to_array(imgVal)
                frames_arrayVal.append(img_arrayVal)
            frames_arrayVal = np.array(frames_arrayVal)
            XVal_batch.append(frames_arrayVal)
            # determine the label based on whether the video contains a fall or not
            if 'non-fall' in video_nameVal:
              yVal_batch.append(0)
            else:
              yVal_batch.append(1)

        XVal_batch = np.array(XVal_batch)
        #print(X_batch)
        yVal_batch = np.array(yVal_batch)
        yield XVal_batch, yVal_batch
