#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing dependencies
get_ipython().system('pip install astroNN')
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from tensorflow import keras
from IPython import display


import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, AveragePooling2D, Flatten, Dense, Dropout, MaxPooling2D
from tensorflow.keras.utils import to_categorical
from keras.preprocessing import image
from keras.callbacks import ReduceLROnPlateau
from keras.optimizers import Adam

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras import utils

from astroNN.datasets import galaxy10
from astroNN.datasets.galaxy10 import galaxy10cls_lookup

from tqdm import tqdm


# In[4]:


# Galaxy10 is a dataset containing 17,736 256x256 pixels colored galaxy images separated in 10
# classes. Galaxy10_DECals.h5 has columns, images, with shape (17,736, 256, 256, 3), ans, ra, dec,
# redshift, and pxscale in unit of arcsecond per pixel.

images, labels = galaxy10.load_data()


# Separating the training set from the test set
x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size = 0.2)

features = ['Disk, Face-on, No Spiral', 'Smooth, Completely round', 'Smooth, in-between round',
           'Smooth, Cigar shaped', 'Disk, Edge-on, Rounded Bulge', 'Disk, Edge-on, Boxy Bulge',
           'Disk, Edge-on, No Bulge', 'Disk, Face-on, Tight Spiral', 
            'Disk, Face-on, Medium Spiral', 'Disk, Face-on, Loose Spiral']
x_train = x_train / 255.0
x_test = x_test / 255.0

x_train.shape, x_test.shape


# In[5]:


# As you can see above, this dataset contains 17,248 colored training images with dimensions
# of 69 by 69 and 4,357 colored testing images with dimensions of 69 by 69.

# The following displays a 10x10 set of images within the dataset
fig = plt.figure(figsize = (20, 20))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.imshow(x_train[i])
    plt.title(features[y_train[i]])
    fig.tight_layout(pad = 3.0)
plt.show()


# In[6]:


# Check Galaxy10 dataset class description along with the matching number of images 
# per category.
df = pd.DataFrame(data = labels)
counts = df.value_counts().sort_index()
print(counts)

def class_distribution(x, y, labels):
    fig, ax = plt.subplots()
    ax.bar(x, y)
    ax.set_xticklabels(labels, rotation = 90)
    plt.show()
    
class_distribution(features, counts, features)


# In[7]:


# As you can see from the above results, the number of images in each class category are extremely
# unbalanced with numbers ranging from 17 (min) *Class 5, Disk, Edge-on, Boxy Bulge*,
# to 6,997 (max) *Class 1, Smooth Completely round*.

# To solve this, we need to consider using other evaluation metrics (used to measure the
# quality of the statistical or machine learning model) since accuracy is no longer a good metric
# for this classification model ("accuracy paradox"), which is due to the imbalance between the
# samples of each class.

# CNN (Convolutional Neural Network) model
# A Sequential model is a linear stack of layers with exactly one input tensor
# and output tensor (generalized vector or matrix of n-dimensions).

model = Sequential()

model.add(Flatten(input_shape = (69, 69, 3)))

model.add(Dense(128, activation = 'relu'))
model.add(Dense(10, activation = 'softmax'))
model_optimizer = Adam(learning_rate = 0.001)
model.compile(optimizer = model_optimizer, loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

model_optimizer = Adam(learning_rate = 0.001)

model.compile(optimizer = model_optimizer, loss = 'sparse_categorical_crossentropy',
metrics = ['accuracy'])
reduceLR = ReduceLROnPlateau(monitor = 'accuracy', factor = 0.001, patience = 1, 
min_delta = 0.01, mode = 'auto')
lol = model.fit(x_train, y_train, epochs = 10, callbacks = [reduceLR])


# In[8]:


# Predictions for this model
predictions = model.predict(x_test)
predict = model.predict(x_test).argmax(axis = 1)

for i in range(10):
    print("Actual: ", features[y_test[i]])
    print("Prediction: ", features[np.argmax(predictions[i])])
    print("______")
    print()


# In[9]:


# Implememented with the LeNet-5 Architecture
# LeNet5 is a small network and contains the basic modules of deep learning: convolutional layer,
# pooling layer, and a full link layer. It is the basis of other deep learning models; moreover,
# the first convolution block in the network consists of two convolutional and average pooling layers
# which are succeeded by a flatten layer and then followed by 3 dense layers.

model2 = Sequential()

# LeNet-5 conv-net architecture
model2.add(Conv2D(filters = 6, kernel_size = (5, 5), strides = (1, 1), activation = 'tanh', input_shape
= (69, 69, 3)))
model2.add(AveragePooling2D(pool_size = (2, 2), strides = (2, 2)))
model2.add(Conv2D(filters = 16, kernel_size = (5, 5), strides = (1, 1), activation = 'tanh'))
model2.add(AveragePooling2D(pool_size = (2, 2), strides = (2, 2)))

model2.add(Flatten())
model2.add(Dense(units = 120, activation = 'tanh'))
model2.add(Dense(units = 84, activation = 'tanh'))
model2.add(Dense(units = 10, activation = 'softmax'))

model_optimizer = Adam(learning_rate = 0.001)
reduceLR = ReduceLROnPlateau(monitor = 'accuracy', factor = 0.001, patience = 1, min_delta = 0.01,
mode = "auto")
model2.compile(optimizer = model_optimizer, loss = 'sparse_categorical_crossentropy', metrics = [
"accuracy"])
model2.fit(x_train, y_train, epochs = 10, callbacks = [reduceLR])


# In[10]:


# Predictions for this model
predictions = model2.predict(x_test)
predict = model2.predict(x_test).argmax(axis = 1)

for i in range(10):
    print("Actual: ", features[y_test[i]])
    print("Prediction: ", features[np.argmax(predictions[i])])
    print("______")
    print()


# In[11]:


# As seen from this data, model 2 has a problem with differentiating 
# classes 7 and 9 (Disk, Face-on, Tight Spiral & Disk, Face-on, Loose Spiral)
# since it's very hard to tell them part. It's also expected that the model
# has trouble classifying class 5 (Disk, Edge-on, Boxy Bulge) due to the low 
# testing and training samples. The model also has a high precision for 
# classes 0 - 4 since they contain the most testing and training samples.

# Therefore, to have a better view of what the model had trouble 
# classifying/differentiating between, we can plot a confusion matrix.

matrix = confusion_matrix(y_test, predict)
sns.heatmap(matrix, annot = True)
plt.title('Galaxy Confusion Matrix')
plt.xlabel('Predicted class')
plt.ylabel('True class')


# In[ ]:


# As seen from the confusion matrix, the model did extremely well
# in classifying classes 0 - 4, especially classes 1 and 2,
# which is what we expected since they had a lot of testing and training data.
# The model also wasn't able to differentiate between classes 5 and 4, 
# 0 and 7, and 8 and 7.


# Conclusion
# Although it's a fairly old CNN, LeNet-5 still holds up, especially for this 
# particular task of image recognition. One of the main problems with this data
# set is that the galaxies are very hard to differentiate between, which is most
# likely why Galaxy Zoo has people who classify an immense amount of Galaxies
# for the sake of improving upon their own model for classification. As expected,
# classes 0 - 4 were the most accurate since they had the most data, and also,
# as expected, it was a struggle to classify classes 7, 9, and 5 since they
# were some of the classes with the least testing and training samples.


# In[ ]:




