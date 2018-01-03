from knn import edit_distance
from math import floor, ceil
from itertools import permutations, chain
from pymining import seqmining



# function to find distance between closest subsequence of a freeman code to candidate
# input: freeman code, candidate sequence
# output: distance
def find_closest_sequence(code, candidate):
    code_length = len(code)
    candidate_length = len(candidate)
    if code_length > candidate_length: #if code is longer than candidate
        # find substring of length candidate_length that minimizes distance
        min_dist = edit_distance(code[0:candidate_length], candidate)
        start_index = 0
        for i in xrange(1,(code_length - candidate_length)+1):
            dist = edit_distance(code[i:i+candidate_length], candidate)
            if dist < min_dist:
                min_dist = dist
                start_index = i
    else:
        min_dist = edit_distance(code, candidate)
        start_index = 0
    stop_index = start_index + candidate_length # stopping index of code substring
    # search stop criterion
    prev_dist = min_dist
    tol = int(floor(float(candidate_length) * .2))   # if distance increases for "tol" straight steps, break loop
    increasing = 0                              #number of consecutive steps that led to increasing distance
    # find length of substring (starting at previously found start_index)that minimizes distance
    # only increase length (decreasing cannot improve distance, include proof in writeup)
    for i in xrange(start_index + candidate_length + 1, code_length+1):
        dist = edit_distance(code[start_index:i],candidate)
        if dist < min_dist:     # if new minimum distance
            min_dist = dist
            stop_index = i
            increasing = 0
        elif dist <= prev_dist: # if distance did not increase
            increasing = 0
        else:                   # if distance increased
            increasing += 1
        if increasing == tol:
            break
        prev_dist = dist
    return min_dist, code[start_index:stop_index]

# function to find frequent sequences from candidates
# input: freeman codes, candidates (all of length K), distance
# distance: how far from sequence is accepted as support
# min_sup_threshold: minimum ratio of support to be determined as frequent
# output: subset of candidates that are frequent in database
def find_frequent(codes, candidates, distance, min_sup_threshold):
    dbsize = len(codes)
    freq_cands = []
    for candidate in candidates:
        support = 0
        for code in codes:
            if find_closest_sequence(code, candidate)[0] <= distance:
                support += 1
        if (float(support) / float(dbsize)) >= min_sup_threshold:
            freq_cands.append(candidate)
    return freq_cands

def generate_new(cands, k):
    if k == 2:
        candidates = list(permutations(cands,2)) + [(x,x) for x in cands]
    else:
        candidates = set(permutations(chain(*cands),k))
    return candidates


# sequence mining algorithm: Generalized Sequential Pattern Mining
# input: freeman codes from one class (digit),
# support threshold to determine frequency: 0 < freq < 1
# output: frequent sequences representative of a class
def modified_GSP(codes, support_threshold):
    candidates = [0,1,2,3,4,5,6,7]
    prev_candidates = []
    k = 2
    while(candidates):
        print k
        prev_candidates = candidates
        candidates = generate_new(candidates, k)
        candidates = find_frequent(codes, candidates, distance=ceil(float(k)/4.0), min_sup_threshold=0.5)
        k += 1
    return prev_candidates


        
codes = ('146372192346464378', '864748586970878437251433152784', '1111182829393844747383')
freq_seqs = seqmining.freq_seq_enum(codes, 2)
print sorted(freq_seqs)

    
    

