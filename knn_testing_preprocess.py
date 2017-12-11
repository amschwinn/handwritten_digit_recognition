# -*- coding: utf-8 -*-
'''
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
'''

#Set working directory
#import os
#os.chdir('D:/GD/MLDM/Machine Learning Project/github')
#os.path.dirname(os.path.abspath(__file__))
#os.getcwd()


#Load prebuilt packages
import timeit
import pandas as pd
import numpy as np
import random
from metric_learn import LMNN
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA

#Load our implementations
import freeman_code as fc
import knn
import load_mnist as lm
import smtplib
import pickle

#%%
#Load mnist dataset
#images,labels,labels_vector = lm.load('Data/train.csv',2000)
images,labels,labels_vector = lm.load('train.csv',2000)

#Separate training and validation sets
train_images = images[0]
train_labels_vect = labels_vector[0]
val_images = images[1]
val_labels_vect = labels_vector[1]

print('load complete')

#%%
#Preprocess and convert into binary images
for i in range(len(train_images)):
    print(i)
    train_image = train_images[i, 0]
    train_images[i, 0] = lm.img_preprocess(train_image)
    #plt.imshow(binary_image, cmap = plt.cm.gray)

print('preprocess complete')
#%%
#Get freeman codes for images
freeman_list = []
for i in range(len(train_images)):
    print(i)
    freeman_list = freeman_list + [fc.freeman_code(train_images[i,0,:,:])]
    
freeman_labels = list(train_labels_vect)
print('freeman encoding complete')

#%%
print "removing outside bayesian eror"
start = timeit.default_timer()
freeman_bay, freeman_labels_bay = knn.remove_outliers_bayesian_error(
        freeman_list,freeman_labels)
end = timeit.default_timer()
bay_1 = (end-start)

print('remove baysian error complete')

#%%
#Remove irrelevant examples
print "removing irrelevant examples"
start = timeit.default_timer()
freeman_irr, freeman_labels_irr = knn.remove_irrelevant(
        freeman_bay,freeman_labels_bay)
end = timeit.default_timer()
irr = (end-start)


#%%
full_preprocess_results = {'freeman_list': freeman_list, 
                           'freeman_labels': freeman_labels,
                           'freeman_bay': freeman_bay, 
                           'freeman_labels_bay':freeman_labels_bay,
                           'freeman_irr': freeman_irr, 
                           'freeman_labels_irr':freeman_labels_irr}

pd.DataFrame.from_dict(full_preprocess_results).to_csv(
        'full_preprocess_results.csv')


