from pymining import freq_seq_enum
from knn import edit_distance
from sklearn.cluster import DBSCAN
from numpy import zeros
import pickle
from numpy import mean

# Returns true if seq1 is a subsequence of seq2
# m is length of str1, n is length of str2
def is_sub_sequence(seq1, seq2):
    m = len(seq1)
    n = len(seq2)
    if m == n:
        return seq1 == seq2
    j = 0 # seq1 index
    i = 0 # seq2 index
    while j < m and i < n:
        if seq1[j] == seq2[i]:
            j = j + 1
        i = i + 1
    return j == m

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

# find representative subsequence for each cluster
# takes subsequence with smallest mean distance to other subseqs in cluster
# input: list of sequences, list of corresponding supports, list of corresponding cluster labels, square distance matrix
# number of sequences for the digit (used when calculating support ratio)
# output: list of subsequences that represent each cluster, corresponding cluster support
def merge_clusters(seqs, sups, cluster_labels, dists, num_seq_digit):
    n = len(sups)
    clusters = set(cluster_labels)
    cluster_seq = []
    cluster_sup = []
    for cluster in clusters:
        cands = []  # candidates to represent this cluster
        for i in range(n):
            # get indices of subsequences in specific cluster
            if cluster_labels[i] == cluster:
                cands.append(i)
        best_avg_dist = 1000
        best_index = 0
        sup = 0  # total support count for all subseqs in cluster
        for cand in cands:
            dist = 0
            for cand2 in cands:
                if cand != cand2:
                    dist += dists[cand, cand2]
            dist = float(dist) / float(len(cands))
            if dist < best_avg_dist:
                best_index = cand
        cluster_seq.append(seqs[best_index])
        cluster_sup.append(round(mean([sups[x] for x in cands]) / float(num_seq_digit), 2))
    return cluster_seq, cluster_sup


def get_freq_sequences(digit):
    if type(digit) == int:
        digit = str(digit)
    f = open('freq_seqs.txt', 'r')
    seq = []
    sup = []
    for line in f:
        tokens = line.split()
        if tokens[0] == digit:
            n = len(tokens)
            for i in range(1,n-3,2):
                seq.append([int(x) for x in tokens[i]])
                sup.append(float(tokens[i+1]))
            return seq, sup


if __name__ == "__main__":

    print get_freq_sequences(4)
    exit(0)
    freeman_train = pickle.load(open('processed_data/freeman_train3.sav','r'))
    freeman_labels = pickle.load(open('processed_data/freeman_labels3.sav','r'))

    digit_codes = {}
    for i in range(len(freeman_labels)):
        if freeman_labels[i] not in digit_codes:
            digit_codes[freeman_labels[i]] = [freeman_train[i]]
        else:
            digit_codes[freeman_labels[i]].append(freeman_train[i])

    digit = 9
    min_sup_thresh = 0.55    # minimum support threshold
    max_gap = 2             # max gap between elements of a subsequence
    min_len_thresh = 0.4    # minimum length threshold (multiplied by mean length of code to find min length for subseqs)
    cluster_max_dist = 2.0  # max dist between two example to be considered same neighborhood (for clustering)
    num_total_seq = len(digit_codes[digit])
    minsup = int(round(len(digit_codes[digit]) * min_sup_thresh))
    min_length = int(round(mean([len(x) for x in digit_codes[digit]]) * min_len_thresh))
    seqs, sup = mine_seqs(digit_codes[digit], minsup=minsup, max_gap=max_gap, min_length=min_length)
    seq_sup = zip(seqs, sup)
    seq_sup.sort(key=lambda x: x[1])
    seqs, sup = zip(*seq_sup)
    dists = pairwise_dists(seqs)
    clusterer = DBSCAN(eps=cluster_max_dist,min_samples=2, metric='precomputed')
    clusterer.fit(dists)
    final_seqs, final_sups = merge_clusters(seqs,sup,clusterer.labels_,dists,num_total_seq)
    print final_seqs
    print final_sups
