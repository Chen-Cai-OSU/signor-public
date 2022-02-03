import collections
from itertools import combinations

import numpy as np

from signor.format.format import pf


def tolist(x):
    if isinstance(x, np.ndarray):
        return x.reshape(-1).tolist()


def expand(lis):
    """ expand a nested list into a non-nest one """
    res = []
    for l in lis:
        if isinstance(l, list):
            res += expand(l)
        else:
            res.append(l)
    return res


def most_freqk(nums, k, precision=3):
    assert isinstance(nums, list)
    return [(pf(item, precision), count) for item, count in collections.Counter(nums).most_common(k)]


def rSubset(arr, r):
    # return list of all subsets of length r
    # to deal with duplicate subsets use
    # set(list(combinations(arr, r))) # https://bit.ly/345tVh7
    return list(combinations(arr, r))


# Driver Function
def sublist(a, indices):
    assert isinstance(a, list)
    return [a[index] for index in indices]

def isflatlist(lis):
    return not any([isinstance(i, list) for i in lis])


if __name__ == '__main__':
    a = list(range(10))
    indices = [1, 3, 7]
    print(sublist(a, indices))

    exit()
    arr = [1, 2, 3, 4]
    r = 2
    print(rSubset(arr, r))

    exit()
    lis = np.random.random((100)).tolist()
    print(most_freqk(lis, 3))
    exit()

    # lis = [[1, 4], [2,3]]
    lis = np.random.random((1, 1, 2, 3)).tolist()

    print(lis)
    print(expand(lis))
