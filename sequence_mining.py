from pymining import freq_seq_enum
from knn import edit_distance
from sklearn.cluster import DBSCAN
from numpy import zeros
import pickle
import numpy as np
from numpy import mean, floor, array
import urllib
from PIL import Image
import base64
import cStringIO

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

# finds the substring of a sequence that is closest to the subsequence
# input: sequence, subsequence
# output: starting index of substring, substring, edit distance from substring to subsequence
def find_match(seq, subsequence):
    n = len(seq)
    m = len(subsequence)
    if n <= m:
        return 0, seq, edit_distance(seq, subsequence)
    best_i = -1
    best_dist = -1
    for i in range(0,n-m):
        dist = edit_distance(seq[i:i+m],subsequence)
        if best_dist == -1 or best_dist > dist:
            best_dist = dist
            best_i = i
    return best_i, seq[best_i:best_i+m], best_dist

# loads frequent sequences for specific digit from text file, returns sequences and supports
# input: digit
# output: frequent sequences, supports
def get_freq_sequences(digit):
    if type(digit) == int:
        digit = str(digit)
    f = urllib.urlopen('https://raw.githubusercontent.com/amschwinn/common_files/master/freq_seqs.txt')
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

# computes new bounding box of subsequence
# input: coordinates for: low row, high row, left column, right column, i (current row) and j (current column
# output: new bounding box (adjusted if current row and column are outside box)
def update_box(lr, hr, lc, rc, i, j):
    if i > lr:
        lr = i
    elif i < hr:
        hr = i
    if j > rc:
        rc = j
    elif j < lc:
        lc = j
    return lr, hr, lc, rc

# draws subsequence on binary grid, returns centered grid
# input: subsequence
# output: grid of 0s and 1s showing centered sequence
def subseq_to_grid(code):
    grid = [[0 for i in range(28)] for j in range(28)] # initialize grid
    i = 13      # current row of grid
    j = 13      # current column of grid
    grid[i][j] = 1
    # bounding box (low row, high row, left col, right col) used for centering at end
    lr = i
    hr = i
    lc = j
    rc = j
    for dir in code:
        if dir == 0 or dir == 1 or dir == 7:
            i -= 1
        if dir <= 3 and dir >= 1:
            j += 1
        if dir >= 3 and dir <= 5:
            i += 1
        if dir >= 5:
            j -= 1
        grid[i][j] = 1
        lr, hr, lc, rc = update_box(lr, hr, lc, rc, i, j)
    # compute center of bounding box
    x = int(floor(float(lr + hr)/2.0))
    y = int(floor(float(lc + rc)/2.0))
    # compute translation parameters
    row_trans = x - 13
    col_trans = y - 13
    # move grid according to translation parameters in order to center the subsequence in the grid
    if row_trans != 0:
        if row_trans > 0:
            del grid[0:row_trans]
            for x in range(row_trans):
                grid.append([0 for z in range(28)])
        else:
            del grid[28 + row_trans:28]
            for x in range(-row_trans):
                grid.insert(0, [0 for z in range(28)])
    if col_trans != 0:
        if col_trans > 0:
            for x in range(28):
                del grid[x][0:col_trans]
            for x in range(28):
                grid[x].extend([0 for x in range(col_trans)])
        else:
            for x in range(28):
                del grid[x][28 + col_trans:28]
            for x in range(28):
                for y in range(-col_trans):
                    grid[x].insert(0, 0)
    return array(grid)


#%%
if __name__ == "__main__":
    digit = '9'
    print('check1')
    seqs, sups = get_freq_sequences(digit)
    grid = subseq_to_grid(seqs[0])
    print('check2')
    import matplotlib.image as mpimg

    mpimg.imsave('freq_seqs/' + digit + '.png', grid * 255, cmap = 'gray')
    exit(0)
    print('check3')
    freeman_train = pickle.load(open('processed_data/freeman_train3.sav','r'))
    freeman_labels = pickle.load(open('processed_data/freeman_labels3.sav','r'))

    digit_codes = {}
    for i in range(len(freeman_labels)):
        if freeman_labels[i] not in digit_codes:
            digit_codes[freeman_labels[i]] = [freeman_train[i]]
        else:
            digit_codes[freeman_labels[i]].append(freeman_train[i])
    print('check4')
    digit = 9
    min_sup_thresh = 0.55    # minimum support threshold
    max_gap = 2             # max gap between elements of a subsequence
    min_len_thresh = 0.4    # minimum length threshold (multiplied by mean length of code to find min length for subseqs)
    cluster_max_dist = 2.0  # max dist between two example to be considered same neighborhood (for clustering)
    num_total_seq = len(digit_codes[digit])
    minsup = int(round(len(digit_codes[digit]) * min_sup_thresh))
    min_length = int(round(mean([len(x) for x in digit_codes[digit]]) * min_len_thresh))
    print('check5')
    seqs, sup = mine_seqs(digit_codes[digit], minsup=minsup, max_gap=max_gap, min_length=min_length)
    seq_sup = zip(seqs, sup)
    seq_sup.sort(key=lambda x: x[1])
    seqs, sup = zip(*seq_sup)
    dists = pairwise_dists(seqs)
    print('check6')
    clusterer = DBSCAN(eps=cluster_max_dist,min_samples=2,metric='precomputed')
    clusterer.fit(dists)
    final_seqs, final_sups = merge_clusters(seqs,sup,clusterer.labels_,dists,num_total_seq)
    print final_seqs
    print final_sups

#%%
def reverse_freeman(row,col,code):
    if code == 0:
        row += -1
        col += 0
    if code == 1:
        row += -1
        col += 1
    if code == 2:
        row += 0
        col += 1
    if code == 3:
        row += 1
        col += 1
    if code == 4:
        row += 1
        col += 0
    if code == 5:
        row += 1
        col += -1
    if code == 6:
        row += 0
        col += -1
    if code == 7:
        row += -1
        col += -1
    return row, col



[ 2, 2, 3, 4, 5, 4, 4, 4, 4, 4, 4, 3, 2, 2, 2, 2, 2, 1, 2, 2 ]
#%%
###############################################################################
#Predict on new test digit
###############################################################################
digit = '2'
seqs, sups = get_freq_sequences(digit)
seq = [2,2,2,3,2,3,2,3,3,2,3,3,2,3,4,3,4,4,4,4,5,5,5,3,2,2,2,1,2,3,4,4,5,6,6,5,6,6,6,6,6,6,6,7,7,0,1,1,1,1,1,1,0,0,7,7,7,6,7,7,6,7,6,7,5,5,6,7,0,0,1]
best_i = [] 
seq_out = [] 
best_dist = []
for subsequence in seqs:
    i, out, dist = find_match(seq, subsequence)
    best_i += [i]
    seq_out += [out] 
    best_dist += [dist]
    
#%%
index = best_dist.index(min(best_dist))
best_i = best_i[index] 
seq_out = seq_out[index] 
best_dist = best_dist[index]



#%%
#For full digit
full = np.zeros((50,50))
row = 20
col = 20
full[row,col] = 1
i = 0
while i < len(seq):
    if i == best_i:
        for code in seq_out:
            row, col = reverse_freeman(row,col,code)
            full[row,col] = 255
            i += 1
    else:       
        row, col = reverse_freeman(row,col,seq[i])
        full[row,col] = 122
        i += 1
#For sub digit
sub = np.zeros((50,50))
row = 20
col = 20
sub[row,col] = 1
i = 0
for code in seq_out:
    row, col = reverse_freeman(row,col,code)
    sub[row,col] = 255
    
#%%
img = full

x_min = np.min((np.sum(img,axis=0) != 0).nonzero())
x_max = np.max((np.sum(img,axis=0) != 0).nonzero())
y_min = np.min((np.sum(img,axis=1) != 0).nonzero())
y_max = np.max((np.sum(img,axis=1) != 0).nonzero())

#Get the bound box size
if (x_max - x_min) > (y_max - y_min):
    bound = (x_max - x_min)
else:
    bound = (y_max - y_min)

full = img[y_min-10:y_max+10,x_min-10:x_max+10]

img = sub

x_min = np.min((np.sum(img,axis=0) != 0).nonzero())
x_max = np.max((np.sum(img,axis=0) != 0).nonzero())
y_min = np.min((np.sum(img,axis=1) != 0).nonzero())
y_max = np.max((np.sum(img,axis=1) != 0).nonzero())

#Get the bound box size
if (x_max - x_min) > (y_max - y_min):
    bound = (x_max - x_min)
else:
    bound = (y_max - y_min)

sub = img[y_min-10:y_max+10,x_min-10:x_max+10]

#%%
#Save the images to data URI
check = Image.fromarray(sub).resize((sub.shape[0]*10,sub.shape[1]*10)).convert('L')
check2 = Image.fromarray(full).resize((full.shape[0]*10,full.shape[1]*10)).convert('L')

buffer1 = cStringIO.StringIO()
check.save(buffer1, format="PNG")
img_str = base64.b64encode(buffer1.getvalue())
img_str = 'data:image/png;base64,'+img_str

buffer2 = cStringIO.StringIO()
check2.save(buffer2, format="PNG")
img_str2 = base64.b64encode(buffer2.getvalue())
img_str2 = 'data:image/png;base64,'+img_str2