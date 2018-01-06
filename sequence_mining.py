from pymining import freq_seq_enum
from knn import edit_distance
from sklearn.cluster import DBSCAN
from numpy import zeros


# creates distance matrix for edit distance between examples
def pairwise_dists(seqs):
    n = len(seqs)
    dists = zeros((n,n))
    for i in range(n-1):
        for j in range(i+1,n):
            distance = edit_distance(seqs[i], seqs[j])
            dists[i,j] = distance
            dists[j,i] = distance
    return dists

# function to change how freeman codes are stored to comply with sequence mining algorithm implementation
# input = list of lists of ints, output = tuple of strings
def format_data(codes):
    n = len(codes)
    for i in range(n):
        new_code = ''
        for dir in codes[i]:
            new_code += str(dir)
        codes[i] = new_code
    codes = tuple(codes)
    return codes

# function that mines frequent subsequences from list of sequences
# input: sequences, min support, max gap between elements in subsequence, min length of subsequence
# output: list of subsequences, list of supports
def mine_seqs(codes, minsup, max_gap, min_length):
    # change list of ints to strings
    codes = format_data(codes)
    # mine freq seqs, freq_seqs contains tuples of (subsequence, support)
    freq_seqs = freq_seq_enum(codes, minsup, max_gap)
    supports = [x[1] for x in freq_seqs]
    freq_seqs = [[int(y) for y in x[0]] for x in freq_seqs]
    # get rid of subsequences that are too short (< minlength)
    n = len(supports)
    for i in range(n-1,-1,-1):
        if len(freq_seqs[i]) < min_length:
            del freq_seqs[i]
            del supports[i]
    return freq_seqs, supports


codes = [[1,4,6,3,7,2,1,9,2,3,4,6,4,6,4,3,7,8],[8,6,4,7,4,8,5,8,6,9,7,0,8,7,8,4,3,7,2,5,1,4,3,3,1,5,2,7,8,4],
         [1,1,1,1,1,8,2,8,2,9,3,9,3,8,4,4,7,4,7,3,8,3]]

seqs, sup = mine_seqs(codes, minsup=2, max_gap=2, min_length=5)

#codes = ('2123', '12234', '23345', '34456')
dists = pairwise_dists(seqs)
clusterer = DBSCAN(eps=2.0,min_samples=2, metric='precomputed')
clusterer.fit(dists)
print clusterer.components_
print clusterer.labels_

    

