from collections import Counter

# input: two lists of integers representing the freeman codes of two images
# output: the edit distance between the two lists
# uses dynamic programming and full m-by-n space table
def edit_distance(im1, im2):
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
                distance[i][j] = (min(distance[i-1][j], distance[i][j-1], distance[i-1][j-1])) + 1
    return distance[m][n]

# input: image to be classified, training features, training labels (same order), k
# output: majority class of k-nearest neighbors to input image
# no optimizations have been used for this algorithm
def knn(im1, X, labels, k):
    distances = []
    for example in X:
        distances.append(edit_distance(im1, example))
    knn_classes = [y for _,y in sorted(zip(distances, labels))[:k]]
    counter = Counter(knn_classes)
    return counter.most_common(4)[0][0]




im1 = [1,2,3]
X = [[1,2,1],[4,5,2],[2,1,2],[1,2,5]]
labels = [1,2,3,1]
print knn(im1,X,labels,2)

