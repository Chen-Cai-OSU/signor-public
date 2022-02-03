""" memory related """

import datetime
import sys

import numpy as np
import torch

from signor.format.format import pf


def test():
    import psutil
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss  # in bytes
    print(mem / (1024 ** 2.0), 'gb')


def du(tail=5):
    # find out largest 5(tail) file under current dir

    cmd = f' du -sh ./* | sort -h | tail -{tail}'
    os.system(cmd)


def metric_conv(size, metric='M'):
    """ convert bytes into other metric """
    if metric == 'B':  # bytes
        return pf(size)
    elif metric == 'K':
        return pf(size / 1024.0)
    elif metric == 'M':
        return pf(size / (1024.0 ** 2))
    elif metric == 'G':
        return pf(size / (1024.0 ** 3))
    else:
        raise Exception(f'Only support metric B/K/M/G. {metric} is not supported.')


def getmem(x, metric="M"):
    size = sys.getsizeof(x)
    size = metric_conv(size, metric=metric)

    return size


def f_size(f="/home/cai.507/Documents/DeepLearning/Signor/signor/graph/stratified.py", metric='M'):
    """ find the size of file f """
    s = os.path.getsize(f)  # bytes
    s = metric_conv(s, metric=metric)

    # modification time
    t = os.path.getmtime(f)
    t = datetime.datetime.fromtimestamp(t).replace(microsecond=0)
    print(f'{f}: {s} {metric}.\nModification time: {t}\n')


import os


def dir_size(dir='.'):
    """ https://gist.github.com/SteveClement/3755572
    """
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(dir):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total_size += os.path.getsize(fp)
    print(f'dir {dir} is {metric_conv(total_size, metric="M")} M.')
    return total_size


def display_top(snapshot, key_type='lineno', limit=10):
    import tracemalloc
    import linecache

    # https://docs.python.org/3/library/tracemalloc.html
    snapshot = snapshot.filter_traces((
        tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
        tracemalloc.Filter(False, "<unknown>"),
    ))
    top_stats = snapshot.statistics(key_type)

    print("Top %s lines" % limit)
    for index, stat in enumerate(top_stats[:limit], 1):
        frame = stat.traceback[0]
        print("#%s: %s:%s: %.1f KiB"
              % (index, frame.filename, frame.lineno, stat.size / 1024))
        line = linecache.getline(frame.filename, frame.lineno).strip()
        if line:
            print('    %s' % line)

    other = top_stats[limit:]
    if other:
        size = sum(stat.size for stat in other)
        print("%s other: %.1f KiB" % (len(other), size / 1024))
    total = sum(stat.size for stat in top_stats)
    print("Total allocated size: %.1f KiB" % (total / 1024))


from sys import getsizeof


def mb(a):
    print(round(getsizeof(a) / 1024 / 1024, 2))


if __name__ == '__main__':
    x = np.random.random((20000, 210))
    mb(x)

    exit()
    dir = '/home/cai.507/Dropbox/Wei_Data/HEA_System/Four_Elements/A_Cai_10th_Mar4/MoVTiCu/fcc/'
    print(dir_size(dir=dir))

    exit()
    f_size()
    exit()
    du()

    x = np.random.random((1000, 1000))
    y = torch.tensor(x)
    for metric in ['B', 'K', 'M', 'G']:
        print(f'Size of x in {metric}: {getmem(x, metric=metric)}')
        print(f'Size of y in {metric}: {getmem(x, metric=metric)}')

    exit()
