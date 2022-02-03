# Created at 2020-10-01
# Summary: genrate combinations, permutations

import random
from itertools import combinations
from signor.utils.random_ import fix_seed

fix_seed()


def exclude_list(sample_ratio=0.1):
    # generate feat selection list
    ret = []
    sum = 0
    for i in range(1, 9):
        out = list(combinations(range(9), i))
        if len(out) > 10:
            out = random.sample(out, min(len(out), int(sample_ratio * len(out)) + 1))

        sum += len(out)
        ret += [','.join(map(str, t)) for t in out]
    ret += ['-1']
    return ret


if __name__ == '__main__':
    ret = exclude_list()
    print(ret)
