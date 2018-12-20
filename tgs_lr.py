
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
print(os.listdir("../input"))
import pdb

from tqdm import tqdm_notebook

# Standard scientific Python imports
import matplotlib.pyplot as plt

# Import datasets, classifiers and performance metrics
from sklearn import svm, metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import load_img

# Loading of train and test dataframes
train_df = pd.read_csv("../input/train.csv", index_col="id", usecols=[0])
depths_df = pd.read_csv("../input/depths.csv", index_col="id")
train_df = train_df.join(depths_df)
test_df = depths_df[~depths_df.index.isin(train_df.index)]

# Loading of images
train_df["images"] = [np.array(load_img("../input/train/images/{}.png".format(idx), grayscale=True)) / 255 for idx in tqdm_notebook(train_df.index)]
train_df["masks"] = [np.array(load_img("../input/train/masks/{}.png".format(idx), grayscale=True)) / 255 for idx in tqdm_notebook(train_df.index)]

# Create label for each image (i.e. if there is salt or not- binary classification)
train_df["label"] = (train_df.masks.map(np.sum)>0).astype(int)

# Reshape
n_samples = len(train_df.images)
salt_img = np.zeros((n_samples, 101, 101))
salt_reshaped = np.zeros((2 * n_samples, 101*101))
for i in range(n_samples):
    salt_img[i] = train_df.images[i]
salt_img = np.append(salt_img, [np.fliplr(x) for x in salt_img], axis = 0)
for i in range(len(salt_img)):
    salt_reshaped[i] = salt_img[i].reshape(-1)
label = np.append(train_df.label, train_df.label)

# Split into train and test set
X_train, X_test, y_train, y_test = train_test_split(salt_reshaped, label, test_size=0.2, shuffle=True)
print('Training data and target sizes: \n{}, {}'.format(X_train.shape,y_train.shape))
print('Test data and target sizes: \n{}, {}'.format(X_test.shape,y_test.shape))

# Create logistic Regression classifier
clr = LogisticRegression(random_state = 1, multi_class = "ovr", solver = 'sag')

# Fit to train data
clr.fit(X_train, y_train)

# Predict the binary class on test data
y_pred = clr.predict(X_test)

print("Classification report for classifier %s:\n%s\n"
      % (clr, metrics.classification_report(y_test, y_pred)))

