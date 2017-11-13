from collections import Counter
from random import shuffle, randrange
from math import floor, sqrt
from numpy import zeros

# input: two lists of integers representing the freeman codes of two images
# output: the edit distance between the two lists
# uses dynamic programming and full m-by-n table
def edit_distance(im1, im2, directions=False):
    m = len(im1)
    n = len(im2)
    # create table for storing results
    distance = [[0 for i in xrange(n+1)] for j in xrange(m+1)]
    for i in xrange(m+1):
        for j in xrange(n+1):
            if i == 0:
                distance[i][j] = j   # add to characters of j if empty list
            elif j == 0:
                distance[i][j] = i   # same if other freeman code is empty
            elif im1[i-1] == im2[j-1]:
                distance[i][j] = distance[i-1][j-1]
            else:
                if directions:
                    distance[i][j] = min(distance[i-1][j] + 1, distance[i][j-1] + 1,
                                         distance[i-1][j-1] + float(direction_cost(im1[i-1],im2[j-1])) / 4.0)
                else:
                    distance[i][j] = (min(distance[i-1][j], distance[i][j-1], distance[i-1][j-1])) + 1
    return distance[m][n]

# function that computes the directional distance between two integers in a freeman code
# computes the distance clockwise and counterclockwise around direction key, returns minimum
# input: two integers representing directions from freeman codes
# output: minimum distance required to change from one direction two another
def direction_cost(x,y):
    direction1 = abs(x-y)
    direction2 = (min(x,y)+ 8) - max(x,y)
    return min(direction1,direction2)

# euclidean distance calculations
# input: two lists (histograms of freeman codes)
# output: distance between two lists
def euclidean_distance(im1, im2):
    total = 0
    for i in xrange(8):
        total += (im1[i] - im2[i]) ^ 2
    return sqrt(total)

# k-nearest Neighbors
# input: image to be classified, training features, training labels (same order), k
# output: majority class of k-nearest neighbors to input image
def knn(im1, X, labels, k, edit = True):
    distances = []
    for example in X:
        if edit:
            distances.append(edit_distance(im1, example))
        else: #euclidean
            distances.append(euclidean_distance(im1, example))
    knn_classes = [y for _,y in sorted(zip(distances, labels))[:k]]
    counter = Counter(knn_classes)
    return counter.most_common(1)[0][0]

# k nearest neighbors with sped-up nieghbor seeking
# input: im1 to be classified, training examples, training labels, k,
# precomputed distances between all training examples,
# whether edit distance should be used (as opposed to euclidean)
# output: class prediction of image
def knn_efficient(im1, X, labels, k, precomputed_distances, edit= True):
    n = len(labels)
    candidates = list(xrange(n))    # all examples are candidates
    lower_bounds = [0 for i in xrange(n)]
    min_dist = float("inf")

    # until no more candidates
    while(candidates):
        best_candidate = min(candidates, key=lambda x: lower_bounds[x]) # select best lower bound
        # compute distance from best candidate to example
        if edit:
            best_cand_dist = edit_distance(im1, X[best_candidate], directions=False)
        else:
            best_cand_dist = euclidean_distance(im1, X[best_candidate])
        # update best so far if closer
        if best_cand_dist < min_dist:
            min_dist = best_cand_dist
            best_index = best_candidate
        # remove candidates whose lower bound is greater than min distance
        old_candidates = candidates
        candidates = []
        for x in old_candidates:
            lower_bound = abs(best_cand_dist - precomputed_distances[best_candidate][x])
            # update lower bound if possible
            lower_bounds[x] = max(lower_bounds[x], lower_bound)
            if lower_bounds[x] < min_dist:
                candidates.append(x)
    return labels[best_index]

# function to precompute distances before efficient knn search
# input: training features, boolean for distance (True = edit distance, False = Euclidean distance)
# output: n by n matrix of distances between all examples in training set
def precompute_distances(X, edit = True):
    n = len(X)
    distance_matrix = zeros((n, n))
    for i in xrange(n):
        for j in xrange(n):
            if distance_matrix[j][i] > 0:
                distance_matrix[i][j] = distance_matrix[j][i]       # use symmetry to make more efficient
            elif not i == j:
                if edit:
                    distance_matrix[i][j] = edit_distance(X[i],X[j],directions=False)
                else:
                    distance_matrix[i][j] = euclidean_distance(X[i],X[j])
    return distance_matrix

# function that removes outliers and examples in the bayesian error region
# input: features, labels of full dataset
# output: features, labels of cleaned dataset
def remove_outliers_bayesian_error(X, labels):
    n = len(labels)
    shuffled_dataset = shuffle(list(zip(X, labels)))
    X, labels = zip(*shuffled_dataset)
    X1 = X[0:floor(n/2)]
    X2 = X[floor(n/2)+1:n]
    labels1 = labels[0:floor(n/2)]
    labels2 = labels[floor(n/2) + 1:n]
    S1_size = len(labels1)
    S2_size = len(labels2)
    prev_S1_size = 0
    prev_S2_size = 0
    while(not S1_size == prev_S1_size and not S2_size == prev_S2_size): #while sets are not stable
        prev_S1_size = S1_size
        prev_S2_size = S2_size
        # classify S1 with S2 and remove misclassified
        for i in range(S1_size,0,-1):# have to go in reverse order to not throw off indices when deleting missclassified
            pred_class = knn(X1[i],X2,labels2,k=1)
            if not pred_class == labels1[i]:
                del labels1[i]
                del X1[i]
        # classify S2 with S1 and remove misclassified
        for i in range(S2_size,0,-1):
            pred_class = knn(X2[i],X1,labels1,k=1)
            if not pred_class == labels2[i]:
                del labels2[i]
                del X2[i]
        S1_size = len(labels1)
        S2_size = len(labels2)
    return X1.extend(X2), labels1.extend(labels2)

# function that removes irrelevant examples from dataset
# input: features, labels of full dataset
# output: features, labels of cleaned dataset
def remove_irrelevant(X, labels):
    # randomly select example to start in storage
    start_index = randrange(0,len(labels))
    storageX = X[start_index]
    del X[start_index]
    storageY = labels[start_index]
    del labels[start_index]
    storage_size = 1
    prev_storage_size = 0
    while(not storage_size == prev_storage_size):
        prev_storage_size = storage_size
        for features, label in zip(X,labels):
            if not knn(features, storageX, storageY, k=1) == label:
                storageX.append(features)
                storageY.append(label)
        storage_size = len(storageY)
    return storageX, storageY


#
# im2 = [2,3,4,5,6,7]
# print edit_distance(im1,im2)
# print edit_distance(im1,im2,directions=True)

im1 = [1,2,3]
X = [[1,2,1],[4,5,2],[2,1,2],[1,2,5]]
precomputed = precompute_distances(X, True)
labels = [1,2,3,1]
print knn_efficient(im1,X,labels,1,precomputed)

