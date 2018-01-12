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
from metric_learn import LMNN
import urllib
import codecs
import pickle

#Load our implementations
import freeman_code as fc
import knn
import load_mnist as lm
import smtplib
import pickle
import trace_boundary as tb

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

#%%
#Preprocess and convert into binary images
def preprocess_convert(train_images):
    for i in range(len(train_images)):
        print(i)
        train_image = train_images[i, 0]
        '''
        #Ensure the edges are padded
        if i in [6107,14895]: 
            train_images[i, 0] = tb.trace_boundary(train_image,threshold=0)
        elif i in [10135,16852]:
            train_images[i, 0] = tb.trace_boundary(train_image,threshold=.28)
        elif i in [15219]:
            train_images[i, 0] = tb.trace_boundary(train_image,threshold=.88)
        elif i in [16811]:
            train_images[i, 0] = tb.trace_boundary(train_image,threshold=.45)
        else:
            train_images[i, 0] = tb.trace_boundary(train_image,threshold=.2)
        '''
        train_images[i, 0] = lm.img_preprocess(train_image)
        #plt.imshow(binary_image, cmap = plt.cm.gray)
    return train_images

#train_images = train_images[:20000,:,:,:]
#val_images = val_images[:1500,:,:,:]


#train_images = preprocess_convert(train_images)
val_images = preprocess_convert(val_images)


print('preprocess complete')
#%%
#Get freeman codes for images
def freeman_code(train_images):
    freeman_list = []
    for i in range(len(train_images)):
        print(i)
        freeman_list = freeman_list + [fc.freeman_code(train_images[i,0,:,:])]
    return freeman_list

#freeman_list = freeman_code(train_images)
freeman_val = freeman_code(val_images)

print('freeman encoding complete')


#%%
#Take subset of freeman codes
freeman_train = []
freeman_labels = []
freeman_index = []

for i in sorted(random.sample(xrange(len(freeman_list)),2000)):
    freeman_train = freeman_train + [freeman_list[i]]
    freeman_labels = freeman_labels + [train_labels_vect[i]]
    freeman_index = freeman_index + [i]
#Save for later
pickle.dump(freeman_train, open('freeman_train4.sav', 'wb'))
pickle.dump(freeman_labels, open('freeman_labels4.sav', 'wb'))

#%%
'''
#Load saved dataset
freeman_train = pickle.load(open('freeman_train.sav','r'))
freeman_labels = pickle.load(open('freeman_labels.sav','r'))
'''
#%%
'''
#Take even smaller subset
freeman_train = freeman_train[0:100]
freeman_labels = freeman_labels[0:100]
'''

#%%
#Remove examples outside of bayseian error
print "removing outside bayesian eror"
start = timeit.default_timer()
freeman_train, freeman_labels = knn.remove_outliers_bayesian_error(freeman_train,freeman_labels)
end = timeit.default_timer()
print(end-start)

#Save for later
pickle.dump(freeman_train, open('freeman_train_bay4.sav', 'wb'))
pickle.dump(freeman_labels, open('freeman_labels_bay4.sav', 'wb'))

#%%
#Remove irrelevant examples
print "removing irrelevant examples"
start = timeit.default_timer()
freeman_train, freeman_labels = knn.remove_irrelevant(freeman_train,freeman_labels)
end = timeit.default_timer()
print(end-start)

pickle.dump(freeman_train, open('freeman_train_irr4.sav', 'wb'))
pickle.dump(freeman_labels, open('freeman_labels_irr4.sav', 'wb'))

#%%
#Save for later
pickle.dump(freeman_train, open('processed_data/freeman_train4_bigger.sav', 'wb'))
pickle.dump(freeman_labels, open('processed_data/freeman_labels4_bigger.sav', 'wb'))
#%%
'''
#Enocde in string for easy access in Exaptive
freeman_total = [freeman_hist, freeman_labels, 1]
pickled = codecs.encode(pickle.dumps(freeman_total), "base64").decode()
pickled = str(pickled)
pickle_file = open("processed_data/freeman_total_hist.txt", "w")
pickle_file.write(pickled)
'''
#%%
#Load saved dataset
#For preprocess 1
freeman_train = pickle.load(open('processed_data/freeman_train_bay.sav','r'))
freeman_labels = pickle.load(open('processed_data/freeman_labels_bay.sav','r'))

'''
#From serialized string
pickled = urllib.urlopen("https://raw.githubusercontent.com/amschwinn/common_files/master/freeman_total.txt").read()
freeman_total = pickle.loads(codecs.decode(pickled.encode(), "base64"))
'''

#%%
#get freeman_histograms
#Calculate frequency of each direction in a freeman code
#Input: freeman code
#Output: count for each direction
def freeman_freq_hist(X):
    amount = np.array(np.repeat(0,8),dtype='float64').reshape((1,8))
    code, counts = np.unique(X, return_counts=True)
    
    for j in range(len(code)):
        amount[0,code[j]] = counts[j]
    
    return amount

#get freeman histograms for training set
freeman_hist = []
for i in freeman_train:
    amount = freeman_freq_hist(i)
    freeman_hist = freeman_hist + amount.tolist()



#%%

#Calculate edit distances
start = timeit.default_timer()
edit_dist = knn.precompute_distances(freeman_train, True)
edit_labels = np.array(freeman_labels)
'''
pd.DataFrame(edit_dist).to_csv('edit_dist_results.csv')

pd.DataFrame(edit_dist).to_csv('edit_dist_labels.csv')
'''
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
'''
#Split train and test
dist_train,dist_val,label_train,label_val,free_train,free_val=train_test_split(
                        edit_dist,edit_labels,freeman_train,train_size=.9)
'''
#%%
'''
hist_train,hist_val,label_train,label_val,free_train,free_val=train_test_split(
                        freeman_hist,hist_labels,freeman_train,train_size=.9)
'''

#%%
'''
#test 1 knn iteration
test = free_train[3]
test_label = label_train[3]
print(test_label)
start = timeit.default_timer()
test_pred = knn.knn(test, free_train, label_train, 1)
print(test_pred)
end = timeit.default_timer()
print(end-start)
'''

#%%
#Test full iteration
pred_label = []
print('knn')
start = timeit.default_timer()
for i in freeman_val:
    pred = knn.knn(i, freeman_train, freeman_labels, 1)
    pred_label = pred_label + [pred]
end = timeit.default_timer()
print(end-start)
#%%
pred_label = np.array(pred_label)
knn_acc = accuracy_score(val_labels_vect,pred_label)
print(knn_acc)

#%%
'''
#Test efficient KNN
test = dist_val[0]
start = timeit.default_timer()
knn.knn_efficient(test,free_train,label_train,1,dist_train)
end = timeit.default_timer()
print(end-start)
'''

#%%
#Verify edit distance
#Should be edit idst of 5
knn.edit_distance(['i','n','t','e','n','t','i','o','n'],
                  ['e','x','e','c','u','t','i','o','n'])

