""" Format related function """

import math
import time

import numpy as np
from colorama import init
from termcolor import colored

init()


def almostequal(x, y, threshold=1e-2):
    if abs(x - y) < threshold:
        return True
    else:
        return False


class delimiter():
    def __init__(self):
        pass

    def small(self):
        print('-' * 10)

    def medium(self):
        print('-' * 50)

    def large(self):
        print('-' * 100)


def print_line(n=150):
    print('-' * n)


import collections


def format_counter(cnt, precision=2):
    if isinstance(cnt, collections.Counter):
        for k, v in cnt.items():
            print(f'{k}: {pf(v, precision=precision)}')
    elif isinstance(cnt, list):
        for (k, v) in cnt:
            print(f'{k}: {pf(v, precision=precision)}')
    else:
        raise NotImplementedError


def pf(nbr, precision=1):
    """ precision format """
    # assert type(nbr)==float
    import torch
    if isinstance(nbr, torch.Tensor):
        nbr = nbr.item()
    if math.isnan(nbr):
        return 0
    else:
        return round(nbr * (10 ** precision)) / (10 ** precision)


def rm_zerocol(data, cor_flag=False, print_flag=False):
    """ remove zero columns """

    # data_ = np.zeros((2,10))
    # data_[1,3] = data_[1,5] = data_[1,7] = 1
    n_col = np.shape(data)[1]
    del_col_idx = np.where(~data.any(axis=0))[0]
    remain_col_idx = set(range(n_col)) - set(del_col_idx)

    x = np.delete(data, np.where(~data.any(axis=0))[0], axis=1)

    if print_flag:
        print('the shape before removing zero columns is %s' % (np.shape(data)))
        print('the shape after removing zero columns is %s' % (np.shape(x)))

    if cor_flag == True:
        correspondence_dict = dict(zip(range(len(remain_col_idx)), remain_col_idx))
        inverse_correspondence_dict = dict(zip(remain_col_idx, range(len(remain_col_idx))))
        return (x, correspondence_dict, inverse_correspondence_dict)
    else:
        return x


def timestamp(seconds=False):
    """ return current date. 2020-02-01_22-32"""
    if seconds:
        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    else:
        timestamp = time.strftime("%Y-%m-%d_%H-%M")
    return timestamp


def filename(f):
    """ give a long filename, convert it to short one.
    '/home/cai.507/Documents/DeepLearning/Signor/signor/mlp_img/image_20.png' to image_20.png
    """

    assert isinstance(f, str)
    if f[-1] == '/':
        f = f[:-1]
    return f.split('/')[-1]


def coordinates(idx=0, n_row=2, n_col=2):
    """ return coordinates of subplot. """

    n_total = n_row * n_col
    assert idx < n_total, f'{idx} should be smaller than n_total {n_total}'
    row_idx = idx // n_col
    col_idx = idx % n_col
    return row_idx, col_idx


def banner(text='', ch='=', length=140, compact=False, v=True, h=False):
    """ http://bit.ly/2vfTDCr
        print a banner
    """
    if not v:
        return
    if h:
        text = red(text)
        length += 9  # to make banner look like the same with the case when h is False

    spaced_text = ' %s ' % text if len(text) > 0 else ''
    banner = spaced_text.center(length, ch)
    print(banner)
    if not compact:
        print()


def args_print(args, one_line=False):
    """ pretty print cmd with lots of args
    /home/cai.507/anaconda3/pretrain/bin/python -u /home/cai.507/Documents/DeepLearning/Signor/signor/graph/cgcnn/code/chem/finetune.py
    --eph           10
    --model_file            /home/cai.507/Dropbox/2020_Spring/Network/proj/data/TianXie/model/mp-ids-3402/band_gap/chem_pretrain/eph_10_epoch_501.pth
    --device                0
    --xt_id                 3402
    --xt_prop               band_gap

    """
    for i in range(20):
        args = args.replace('  ', ' ')

    arglis = args.split(' ')
    new_arglist = []
    for i, token in enumerate(arglis):
        if '--' in token:
            token = '\n' + token
        elif token in ['-u', 'nohup']:
            pass
        elif '.py' in token:
            pass
        elif 'python' in token:
            pass
        else:
            space = (30 - len(arglis[i - 1])) * ' '
            token = space + token  # '{:>35}'.format(token) #
        new_arglist.append(token)

    newargs = ' '.join(new_arglist) + '\n'

    if not one_line:
        print(newargs)
    else:
        newargs = one_liner(newargs)
        print(newargs)


def one_liner(cmd):
    """ convert cmd that takes many lines into just one line """
    assert isinstance(cmd, str)
    cmd = cmd.replace('\n', '')
    for _ in range(10):
        cmd = cmd.replace('  ', ' ')
    return cmd

# @deprecated
# def red(x):
#     return colored(x, "red")

def red(*args):
    ret = []
    for x in args:
        ret.append(colored(x, "red"))
    return ' '.join(ret)

def print_v(x, v=True, **kwargs):
    """

    :param x: content
    :param v: verbose flag
    :return:
    """
    if v == True:
        print(x, **kwargs)


def full_print(df):
    # https://bit.ly/34hbeYy
    import pandas as pd
    with pd.option_context('display.max_rows', None):  # more options can be specified also
        print(df)

def value_to_float(x):
    # https://bit.ly/3sDqLxe
    if type(x) == float or type(x) == int:
        return x
    if 'K' in x:
        if len(x) > 1:
            return float(x.replace('K', '')) * 1000
        return 1000.0
    if 'M' in x:
        if len(x) > 1:
            return float(x.replace('M', '')) * 1000000
        return 1000000.0
    if 'B' in x:
        return float(x.replace('B', '')) * 1000000000
    return 0.0

def float_to_value(x, n_digits=1):
    if isinstance(x, str):
        try:
            x = float(x)
        except ValueError:
            return x
    if x < 1e3:
        return x
    elif x < 1e6:
        return str(pf(x/1e3, n_digits)) + 'K'
    elif x < 1e9:
        return str(pf(x/1e6, n_digits)) + 'M'
    elif x < 1e12:
        return str(pf(x / 1e9, n_digits)) + 'B'
    else:
        raise NotImplementedError

def time_format(x, n_digits=1):
    if isinstance(x, str):
        x = float(x)
    if x < 60:
        return str(pf(x, n_digits)) + 's'
    elif x < 3600:
        return str(pf(x/60, n_digits)) + 'm'
    elif x < 3600 * 24:
        return str(pf(x/(3600), n_digits)) + 'h'
    else:
        return str(pf(x/(3600*24), n_digits)) + 'd'

if __name__ == '__main__':
    import pandas as pd
    df = pd.DataFrame({'col': ['1.234M', '1.001B']})
    print(df)
    df['col'] = df['col'].apply(value_to_float)
    print(df)
    df['col'] = df['col'].apply(float_to_value)
    print(df)
    exit()
    from collections import Counter

    d = Counter({'a': 1e-5, 'b': 1 + 1e-6})
    format_counter(d, precision=5)

    banner('hello', h=True)
    banner('hello', h=False)
    banner('hello-abcd', h=True)
    exit()

    print(timestamp(seconds=True))

    exit()
    print(red({1: 2, 'a': 'b'}))
    print(red(1))
    exit()

    args = '/home/cai.507/anaconda3/envs/pretrain/bin/python -u  /home/cai.507/Documents/DeepLearning/Signor/signor/graph/cgcnn/code/finetune.py   --model_fileabcdfe /home/cai.507/Dropbox/2020_Spring/Network/proj/data/TianXie/model/mp-ids-3402/band_gap/model_epochs_501_bucket_True_eph_21.pth --dataset cif_new --batch_size 128 --epochs 20 --split random --filename  test_pretrain_epochs_501_bucket_True_eph_21_test_epochs_20 --device 0 --xt_id 3402 --xt_prop band_gap   --bucket '
    args_print(args, one_line=True)
    exit()

    num = 1 / 3.0
    print(pf(num, precision=10))
    exit()

    print(coordinates(idx=7, n_row=3, n_col=4))
    exit()

    print(pf(0.00001))
    print(pf(1.234))
    exit()

    print(timestamp())
    print_line()
