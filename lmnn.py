# -*- coding: utf-8 -*-
"""
Implimenting LMNN metric learning
to move observations of same class closer together
while separating from different classes.

Using LMNN package from metric-learn python library.
This implimentation is based closely on research paper
Distance Metric Learning for Large Margin Nearest Neighbor 
Classification Kilian Q. Weinberger, John Blitzer, Lawrence K. Saul

https://all-umass.github.io/metric-learn/metric_learn.lmnn.html

MLDM M2 Machine Learning Project
Handwritten Digit Classification

Austin Schwinn, Joe Renner, Dimitris Tsolakidis
November 24, 2017
"""

#Set working directory
import os
os.chdir('C:\Users\schwi\Google Drive\MLDM\Machine Learning Project\github')
#os.path.dirname(os.path.abspath(__file__))
os.getcwd()

#Load prebuilt packages
import numpy as np
from metric_learn import LMNN
from sklearn.cluster import KMeans

#Load our implimentations
import freeman_code as fc
import knn
import load_mnist as lm

#Load mnist dataset
images,labels,labels_vector = lm.load('Data/train.csv',2000)

#Separate training and validation sets
train_images = images[0]
train_labels_vect = labels_vector[0]
val_images = images[1]
val_labels_vect = labels_vector[1]

#Preprocess and convert into binary images
for i in range(len(train_images)):
    print(i)
    train_image = train_images[i, 0]
    train_images[i, 0] = lm.img_preprocess(train_image)
    #plt.imshow(binary_image, cmap = plt.cm.gray)

#Get freeman codes for images
freeman_train = []
for i in range(len(train_images)):
    print(i)
    freeman_train = freeman_train + [fc.freeman_code(train_images[i,0,:,:])]

#Calculate edit distances
train_dist = knn.precompute_distances(freeman_train, True)

#%%
#Impliment LMNN with edit distance
lmnn = LMNN(k=10, learn_rate=1e-6).fit(train_dist, train_labels_vect)

#Transform into new feature space
train_dist_metric = lmnn.transform(train_dist)

#Run KMeans on LMNN transformed features
kmeans = KMeans(n_clusters=10).fit(train_dist_metric)
