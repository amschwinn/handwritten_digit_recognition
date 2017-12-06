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
images,labels,labels_vector = lm.load('Data/train.csv',2000)
#images,labels,labels_vector = lm.load('train.csv',2000)

#Separate training and validation sets
train_images = images[0]
train_labels_vect = labels_vector[0]
val_images = images[1]
val_labels_vect = labels_vector[1]

print('load complete')

#Preprocess and convert into binary images
for i in range(len(train_images)):
    print(i)
    train_image = train_images[i, 0]
    train_images[i, 0] = lm.img_preprocess(train_image)
    #plt.imshow(binary_image, cmap = plt.cm.gray)

print('preprocess complete')

#Get freeman codes for images
freeman_list = []
for i in range(len(train_images)):
    print(i)
    freeman_list = freeman_list + [fc.freeman_code(train_images[i,0,:,:])]

print('freeman encoding complete')


#%%
#Take subset of freeman codes
freeman_train = []
freeman_labels = []
freeman_index = []

for i in sorted(random.sample(xrange(len(freeman_list)),1000)):
    freeman_train = freeman_train + [freeman_list[i]]
    freeman_labels = freeman_labels + [train_labels_vect[i]]
    freeman_index = freeman_index + [i]
#%%
#Save for later
pickle.dump(freeman_train, open('freeman_train.sav', 'wb'))
pickle.dump(freeman_labels, open('freeman_labels.sav', 'wb'))

#%%
#Load saved dataset
freeman_train = pickle.load(open('freeman_train.sav','r'))
freeman_labels = pickle.load(open('freeman_labels.sav','r'))

#%%
#Take even smaller subset
freeman_train = freeman_train[0:100]
freeman_labels = freeman_labels[0:100]

#%%
#Remove examples outside of bayseian error
print "starting bayesian"
freeman_train, freeman_labels = knn.remove_outliers_bayesian_error(freeman_train,freeman_labels)

#Remove irrelevant examples
print "removing irrelevant examples"
freeman_train, freeman_labels = knn.remove_irrelevant(freeman_train,freeman_labels)
exit(0)
#%%
'''
from random import shuffle
check2=zip(freeman_train,freeman_labels)
check=zip(freeman_train,freeman_labels)
shuffle(check2)
print(check[0])
print(check2[0])
print(check==check2)
'''
#%%
#get freeman_histograms
freeman_hist = np.array(np.repeat(0,8),dtype='float64').reshape((1,8))

for i in freeman_train:
    amount = np.array(np.repeat(0,8),dtype='float64').reshape((1,8))
    code, counts = np.unique(i, return_counts=True)
    for j in range(len(code)):
        amount[0,code[j]] = counts[j]
    freeman_hist = np.vstack((freeman_hist,amount))

freeman_hist = freeman_hist[1:,:]
hist_labels = np.array(freeman_labels)


#%%

#Calculate edit distances
start = timeit.default_timer()
edit_dist = knn.precompute_distances(freeman_train, True)
edit_labels = np.array(freeman_labels)

pd.DataFrame(edit_dist).to_csv('edit_dist_results.csv')

pd.DataFrame(edit_dist).to_csv('edit_dist_labels.csv')

end = timeit.default_timer()
print(end-start)
print('edit distance complete')

#%%
'''
#Run PCA on edit distances
pca = PCA(n_components=200).fit(edit_dist)
print(pca.explained_variance_ratio_.sum())
edit_dist_pca = pca.transform(edit_dist)
'''
#%%
#Split train and test
dist_train,dist_val,label_train,label_val,free_train,free_val=train_test_split(
                        edit_dist,edit_labels,freeman_train,train_size=.9)
#%%
hist_train,hist_val,label_train,label_val,free_train,free_val=train_test_split(
                        freeman_hist,hist_labels,freeman_train,train_size=.9)


#%%
#test 1 knn iteration
test = free_train[0]
test_label = label_train[0]
print(test_label)
start = timeit.default_timer()
test_pred = knn.knn(test, free_train, label_train, 1)
print(test_pred)
end = timeit.default_timer()
print(end-start)

#%%
#Test full iteration
pred_label = []
start = timeit.default_timer()
for i in free_val:
    pred = knn.knn(i, free_train, label_train, 1)
    pred_label = pred_label + [pred]
end = timeit.default_timer()
print(end-start)
#%%
pred_label = np.array(pred_label)
knn_acc = accuracy_score(label_val,pred_label)
print(knn_acc)

#%%
#Test efficient KNN
test = dist_val[0]
start = timeit.default_timer()
knn.knn_efficient(test,free_train,label_train,1,dist_train)
end = timeit.default_timer()
print(end-start)

#%%
#Verify edit distance
#Should be edit idst of 5
knn.edit_distance(['i','n','t','e','n','t','i','o','n'],
                  ['e','x','e','c','u','t','i','o','n'])

