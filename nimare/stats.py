"""Various statistical helper functions"""

import numpy as np


def pearson(x, y):
    """ Correlates row vector x with each row vector in 2D array y. """
    data = np.vstack((x, y))
    ms = data.mean(axis=1)[(slice(None, None, None), None)]
    datam = data - ms
    datass = np.sqrt(np.sum(datam**2, axis=1))
    temp = np.dot(datam[1:], datam[0].T)
    rs = temp / (datass[1:] * datass[0])
    return rs


def fdr(p, q=.05):
    """ Determine FDR threshold given a p value array and desired false
    discovery rate q. """
    s = np.sort(p)
    nvox = p.shape[0]
    null = np.array(range(1, nvox + 1), dtype='float') * q / nvox
    below = np.where(s <= null)[0]
    return s[max(below)] if any(below) else -1
