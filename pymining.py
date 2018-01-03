# modified code from https://github.com/bartdag/pymining/blob/master/pymining/seqmining.py
from collections import defaultdict


def freq_seq_enum(sequences, min_support, max_gap):
    '''Enumerates all frequent sequences.
       :param sequences: A sequence of sequences.
       :param min_support: The minimal support of a set to be included.
       :rtype: A set of (frequent_sequence, support).
    '''
    freq_seqs = set()
    newseq = []
    for seq in sequences:
        newseq.append([seq])
    _freq_seq(newseq, tuple(), 0, min_support, freq_seqs, max_gap, start=True)
    return freq_seqs


def _freq_seq(sdb, prefix, prefix_support, min_support, freq_seqs, max_gap, start=False):
    if prefix:
        freq_seqs.add((prefix, prefix_support))
    locally_frequents = _local_freq_items(sdb, prefix, min_support, start, max_gap)
    if not locally_frequents:
        return
    for (item, support) in locally_frequents:
        new_prefix = prefix + (item,)
        new_sdb = _project(sdb, new_prefix, max_gap, start)
        _freq_seq(new_sdb, new_prefix, support, min_support, freq_seqs, max_gap)


def _local_freq_items(sdb, prefix, min_support, start, max_gap):
    items = defaultdict(int)
    freq_items = []
    for entry in sdb:
        visited = set()
        for subseq in entry:
            j = 0
            for element in subseq:
                if element not in visited:
                    items[element] += 1
                    visited.add(element)
                j += 1
                if not start:
                    if j > max_gap:
                        break
    for item in items:
        support = items[item]
        if support >= min_support:
            freq_items.append((item, support))
    return freq_items


def _project(sdb, prefix, max_gap, start):
    new_sdb = []
    if not prefix:
        return sdb
    current_prefix_item = prefix[-1]
    for entry in sdb:
        entry_proj = []
        for subseq in entry:
            j = 0
            for item in subseq:
                if item == current_prefix_item:
                    projection = subseq[j + 1:]
                    if projection:
                        entry_proj.append(projection)
                j += 1
                if not start:
                    if j > max_gap:
                        break
        if entry_proj:
            new_sdb.append(entry_proj)
    return new_sdb