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
import codecs
import urllib
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
        train_images[i, 0] = lm.img_preprocess3(train_image)
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
#Load saved dataset
#Load saved dataset
#For preprocess 1
freeman_train = pickle.load(open('processed_data/freeman_train_irr3.sav','r'))
freeman_labels = pickle.load(open('processed_data/freeman_labels_irr3.sav','r'))

#%%
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
    
freeman_hist_val = []
for i in freeman_val:
    amount = freeman_freq_hist(i)
    freeman_hist_val  += amount.tolist()

#%%
#Calculate edit distances
start = timeit.default_timer()
edit_dist = knn.precompute_distances(freeman_train, True)
edit_labels = np.array(freeman_labels)

#pd.DataFrame(edit_dist).to_csv('edit_dist_results.csv')

#pd.DataFrame(edit_dist).to_csv('edit_dist_labels.csv')

end = timeit.default_timer()
print(end-start)
print('edit distance complete')

#%%
#Save edit distance
#Enocde in string for easy access in Exaptive
pickled = codecs.encode(pickle.dumps(edit_dist), "base64").decode()
pickled = str(pickled)
pickle_file = open("processed_data/edit_dist_train.txt", "w")
pickle_file.write(pickled)

#%%
'''
#Run PCA on edit distances
pca = PCA(n_components=200).fit(edit_dist)
print(pca.explained_variance_ratio_.sum())
edit_dist_pca = pca.transform(edit_dist)

#Split train and test
dist_train, dist_val, label_train, label_val = train_test_split(edit_dist_pca,
                                                    edit_labels,train_size=.9)
hist_train, hist_val, label_train, label_val = train_test_split(freeman_hist,
                                                    hist_labels,train_size=.9)

#Impliment LMNN with edit distance
start = timeit.default_timer()
lmnn_dist = LMNN(k=3, learn_rate=1e-6).fit(dist_train, label_train)
lmnn_hist = LMNN(k=3, learn_rate=1e-6).fit(hist_train, label_train)

#Transform into new feature space
dist_train_metric = lmnn_dist.transform(dist_train)
dist_val_metric = lmnn_dist.transform(dist_val)
hist_train_metric = lmnn_hist.transform(hist_train)
hist_val_metric = lmnn_hist.transform(hist_val)


end = timeit.default_timer()
print(end-start)
print('lmnn complete')

#Run KNN on LMNN transformed features
start = timeit.default_timer()

KNN_dist = KNeighborsClassifier(n_neighbors=1).fit(dist_train_metric, label_train)
dist_predict = KNN_dist.predict(dist_val_metric)
lmnn_hist_acc = accuracy_score(label_val,dist_predict)

KNN_hist = KNeighborsClassifier(n_neighbors=1).fit(hist_train_metric, label_train)
hist_predict = KNN_hist.predict(hist_val_metric)
lmnn_hist_acc = accuracy_score(label_val,hist_predict)

end = timeit.default_timer()
print(end-start)
'''

#%%
#LMNN using freeman codes and edit distances or freeman histograms
#inputs: new freeman code, training edit dists or freeman hists, 
#   training labels, k,  lmnn model
#   metric refers to edit distance or freeman code histogram (dist or hist)
#output: predicted label for new freeman code
def lmnn(testX, X, y, k, metric='dist'):
    if metric == 'dist':
        #Calculate edit distance between new observation and training examples
        dists = []
        for x in X:
            dist = knn.precompute_distances([testX,x], edit=True)
            dists = dists + [dist[0,1]]
        testX = np.array(dists).reshape((1,len(dists)))
        #Load LMNN Model
        pickled = urllib.urlopen("https://raw.githubusercontent.com/amschwinn/common_files/master/edit_dist_train.txt").read()
        X = pickle.loads(codecs.decode(pickled.encode(), "base64"))
        pickled = urllib.urlopen("https://raw.githubusercontent.com/amschwinn/common_files/master/lmnn_dist.txt").read()
        lmnn_dist = pickle.loads(codecs.decode(pickled.encode(), "base64"))
        lmnn_model = lmnn_dist[0]

    elif metric == 'hist':
        #Calculate frequency of each freeman digit
        testX = np.array(freeman_freq_hist(testX))
        #load lmnn model
        pickled = urllib.urlopen("https://raw.githubusercontent.com/amschwinn/common_files/master/lmnn_hist.txt").read()
        lmnn_hist = pickle.loads(codecs.decode(pickled.encode(), "base64"))
        lmnn_model = lmnn_hist[0]
        #lmnn_model = lmnn_hist
    
    #Use LMNN for metric learning
    X_lmnn = lmnn_model.transform(X)
    testX_lmnn = lmnn_model.transform(testX)
  
    #Predict on transformed dataset using KNN with euclidean distance
    pred = knn.knn(testX_lmnn.tolist()[0], X_lmnn.tolist(), y, k=k, edit=False)
    
    #print(X_lmnn)
    #print(testX_lmnn)
    return pred

#%%
#train LMNN model
freeman_hist = np.array(freeman_hist)
freeman_labels =  np.array(freeman_labels)
print('hist')
start = timeit.default_timer()
lmnn_hist = LMNN(k=5, learn_rate=1e-6).fit(freeman_hist, freeman_labels)
end = timeit.default_timer()
print('hist train time')
print(end-start)
'''
print('dist')
start = timeit.default_timer()
lmnn_dist = LMNN(k=5, learn_rate=1e-6).fit(edit_dist, freeman_labels)
end = timeit.default_timer()
print('edit train time')
print(end-start)
'''
#%%
#Test full iteration
pred_label_edit = []
print('edit lmnn')
start = timeit.default_timer()
it = 0
for i in freeman_val:
    pred = lmnn(i,freeman_train,freeman_labels, k=1, metric='dist')
    pred_label_edit += [pred]
    print(it)
    it += 1
end = timeit.default_timer()
print('edit lmnn run')
print(end-start)

pred_label_edit = np.array(pred_label_edit)
lmnn_acc_edit = accuracy_score(val_labels_vect,pred_label_edit)
print('edit lmnn acc')
print(lmnn_acc_edit)

#%%
#Test full iteration
pred_label_hist = []
print('hist lmnn')
start = timeit.default_timer()
it = 0
for i in freeman_val:
    pred = lmnn(i,np.array(freeman_hist),freeman_labels, k=1, metric='hist')
    pred_label_hist += [pred]
    print(it)
    it += 1
end = timeit.default_timer()
print('hist lmnn run')
print(end-start)

pred_label_hist = np.array(pred_label_hist)
lmnn_acc_hist = accuracy_score(val_labels_vect,pred_label_hist)
print('hist lmnn acc')
print(lmnn_acc_hist)
