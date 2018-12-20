import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
print(os.listdir("./input"))
import pdb

from tqdm import tqdm_notebook

# Standard scientific Python imports
import matplotlib.pyplot as plt

# Import datasets, classifiers and performance metrics
from sklearn import svm, metrics
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import load_img

# Loading of train and test dataframes
train_df = pd.read_csv("./input/train.csv", index_col="id", usecols=[0])
depths_df = pd.read_csv("./input/depths.csv", index_col="id")
train_df = train_df.join(depths_df)
test_df = depths_df[~depths_df.index.isin(train_df.index)]

# Loading of images
train_df["images"] = [np.array(load_img("./input/train/images/{}.png".format(idx), color_mode="grayscale")) / 255 for idx in tqdm_notebook(train_df.index)]
train_df["masks"] = [np.array(load_img("./input/train/masks/{}.png".format(idx), color_mode="grayscale")) / 255 for idx in tqdm_notebook(train_df.index)]

# Create label for each image (i.e. if there is salt or not- binary classification)
train_df["label"] = (train_df.masks.map(np.sum)>0).astype(int)

# Reshape
n_samples = len(train_df.images)
salt_img = np.zeros((n_samples, 101, 101))
for i in range(n_samples):
    salt_img[i] = train_df.images[i]
salt_img = np.append(salt_img, [np.fliplr(x) for x in salt_img], axis = 0)
salt_img = np.expand_dims(salt_img, axis=1)
label = np.append(train_df.label, train_df.label)

# Split into train and test set
X_train, X_test, y_train, y_test = train_test_split(salt_img, label, test_size=0.2, shuffle=True)
print('Training data and target sizes: \n{}, {}'.format(X_train.shape,y_train.shape))
print('Test data and target sizes: \n{}, {}'.format(X_test.shape,y_test.shape))

# Import Keras library
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout

# Create CNN classifier
classifier = Sequential()
classifier.add(Conv2D(16, (3,3), input_shape=(1, 101,101), activation="relu", padding="same", data_format="channels_first"))
classifier.add(MaxPooling2D(pool_size=(2,2)))
classifier.add(Dropout(0.5))
classifier.add(Conv2D(32, (3,3), activation="relu", padding="same"))
classifier.add(MaxPooling2D(pool_size=(2,2)))
classifier.add(Dropout(0.5))
classifier.add(Conv2D(64, (3,3), activation="relu", padding="same"))
classifier.add(MaxPooling2D(pool_size=(2,2)))
classifier.add(Dropout(0.5))
classifier.add(Conv2D(128, (3,3), activation="relu", padding="same"))
classifier.add(MaxPooling2D(pool_size=(2,2)))
classifier.add(Dropout(0.5))
classifier.add(Flatten())
classifier.add(Dense(units=128, activation="relu"))
classifier.add(Dense(units=1, activation="sigmoid"))

# Compile CNN model
classifier.compile(optimizer="adam", loss="binary_crossentropy", metrics=['accuracy'])

classifier.fit(X_train, y_train, batch_size=200, epochs=5, verbose=1, validation_split=0.1, shuffle=True)

pred = classifier.predict(X_test)[:,0]
for i in range(len(pred)):
    if pred[i] == y_test[i]:
        correct_count += 1
accuracy = correct_count / pred.size
print(str(accuracy))
