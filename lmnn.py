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
import os
os.chdir('D:/GD/MLDM/Machine Learning Project/github')
#os.path.dirname(os.path.abspath(__file__))
os.getcwd()


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

#Load our implimentations
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

for i in sorted(random.sample(xrange(len(freeman_list)),20000)):
    freeman_train = freeman_train + [freeman_list[i]]
    freeman_labels = freeman_labels + [train_labels_vect[i]]
    freeman_index = freeman_index + [i]

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
'''
#Calculate edit distances
start = timeit.default_timer()
edit_dist = knn.precompute_distances(freeman_train, True)
edit_labels = np.array(freeman_labels)

pd.DataFrame(edit_dist).to_csv('edit_dist_results.csv')

pd.DataFrame(edit_dist).to_csv('edit_dist_labels.csv')

end = timeit.default_timer()
print(end-start)
print('edit distance complete')
'''
#%%
'''
#Run PCA on edit distances
pca = PCA(n_components=200).fit(edit_dist)
print(pca.explained_variance_ratio_.sum())
edit_dist_pca = pca.transform(edit_dist)
'''
#%%
#Split train and test
#dist_train, dist_val, label_train, label_val = train_test_split(edit_dist_pca,
#                                                    edit_labels,train_size=.9)
hist_train, hist_val, label_train, label_val = train_test_split(freeman_hist,
                                                    hist_labels,train_size=.9)

#Impliment LMNN with edit distance
start = timeit.default_timer()
#lmnn_dist = LMNN(k=3, learn_rate=1e-6).fit(dist_train, label_train)
lmnn_hist = LMNN(k=3, learn_rate=1e-6).fit(hist_train, label_train)

#Transform into new feature space
#dist_train_metric = lmnn_dist.transform(dist_train)
#dist_val_metric = lmnn_dist.transform(dist_val)
hist_train_metric = lmnn_hist.transform(hist_train)
hist_val_metric = lmnn_hist.transform(hist_val)


end = timeit.default_timer()
print(end-start)
print('lmnn complete')

#Run KNN on LMNN transformed features
start = timeit.default_timer()
KNN = KNeighborsClassifier(n_neighbors=1).fit(hist_train_metric, label_train)
label_predict = KNN.predict(hist_val_metric)
lmnn_acc = accuracy_score(label_val,label_predict)

end = timeit.default_timer()
print(end-start)

#%%
#Save models
filename = 'lmnn_model.sav'
pickle.dump(lmnn_hist, open(filename, 'wb'))
filename = 'KNN_model.sav'
pickle.dump(KNN, open(filename, 'wb'))

#%%
###############################################################################
# Other
###############################################################################
'''
#Send email notification when test is complete
server = smtplib.SMTP('smtp.gmail.com', 587)
server.connect("smtp.gmail.com",587)
server.ehlo()
server.starttls()
server.login("schwinnteriscoming@gmail.com", "@lp4aCat")
msg = 'Subject: LMNN Test Complete'
server.sendmail("schwinnteriscoming@gmail.com", "schwinnam@gmail.com", msg)
'''
