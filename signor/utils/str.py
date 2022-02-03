import argparse
from collections import OrderedDict, defaultdict
from pprint import pprint

lis = ['test_pretrain_epochs_501_eph_20_test_epochs_50', 'test_pretrain_epochs_501_eph_201_test_epochs_50']


class sort_model_name():
    """ given a list of  names such as lis (above), sort acoording to a field, such as eph. """

    def __init__(self, list, field='eph', delimiter='_'):
        self.list = list
        self.field = field
        self.delimiter = delimiter
        self.idx = self.field_idx()
        self.field_vals = [float(self.str2lis(lis)[self.idx + 1]) for lis in self.list]

    def field_idx(self):
        s = self.list[0]
        idx = self.str2lis(s).index(self.field)
        return idx

    def str2lis(self, s):
        return s.split(self.delimiter)

    def check_format(self):
        """ make sure all strings has same number o delimiters """
        n = self.list[0].count(self.delimiter)
        for s in self.list:
            assert s.count(self.delimiter) == n, f'{s} has different number of {self.delimiter}s. Expect {n}'

    def sort_by_field(self):
        x = OrderedDict(zip(self.field_vals, self.list))
        sort_keys = sorted(self.field_vals)
        vals = []
        for key in sort_keys:
            vals.append(x[key])
        return vals


def shell2script(s):
    assert isinstance(s, str)
    for _ in range(5):
        s = s.replace('  ', ' ')
    s = s.split(' ')
    print(s)
    return s


def camel_case_split(s):
    """ modified from https://bit.ly/2QV97TZ"""
    assert isinstance(s, str)
    words = [[s[0]]]

    for c in s[1:]:
        if c.isupper():  # words[-1][-1].islower() and c.isupper():
            words.append(list(c))
        else:
            words[-1].append(c)

    return [''.join(word) for word in words]


def arg2str(args):
    from signor.configs.util import dict2name
    assert isinstance(args, argparse.Namespace)
    d = vars(args)  # https://bit.ly/3cX0WPT
    ret = dict2name(d)
    ret = ret.replace('/', '')  # remove / since it will cause trouble when save
    return ret


def hasany(s, s_list):
    """
    :param s: a string
    :param s_list: a list of str
    :return:
    """
    return any(ele in s for ele in s_list)


def slicestr(s, f=None, t=None):
    """
    :param s: a string
    :param f: from
    :param t: to
    :return:
    """
    from_idx = s.index(f)
    to_idx = s.index(t)
    return s[from_idx:to_idx]

    # exit()
    # s = 'error_eigenvalue: array (float64) of shape    (40,)     Nan ratio:      0.0.     0.672(mean)      0.0(min)    2.809(max)    0.554(median)    0.407(std)     40.0(unique) '
    # from_idx = s.index('Nan ratio')
    # to_idx = s.index('(mean')
    # print(s[from_idx:to_idx][-5:])
    #
    # exit()
    # s = 'INFO:root:Idx 0-Epoch: 50. Train(1.7): 0.9(0.265) / 1.0(0.292) / 2.9. Eigenloss: 0.212. Subspaceloss: 0.0'
    # from_idx = s.index('Eigenloss')
    # to_idx = s.index('Subspaceloss')
    # print(s[from_idx:to_idx].split(':'))


def filter_logtime(lis):
    """
    :param lis: a list of strings
    :return: a defaultdict dict of form  {'train': 11.73926, 'test': 2.3}
    """
    import re
    pattern = re.compile("\w*\: \d*\.\d*s")
    total_time = defaultdict(int)
    for string in lis:
        m = pattern.match(string)
        if m is None: continue
        func, t = m.group().split(':')
        t = t.lstrip().rstrip('s')
        total_time[func] += float(t)
    for k, v in total_time.items():
        total_time[k] = int(v)
    return dict(total_time)

import numpy as np
def filter_tqdm(lis, verbose=False):
    """ from tqdm output get iteration time per iteration """
    # todo: refactor
    import re


    #  80%|████████  | 4/5 [03:58<00:59, 59.31s/it]Epoch: 005, Train Loss: 0.303, Test RMSEN loss: 7.868/0.16787654
    pattern = re.compile(".*\[.*\]")

    ret = {'tqdm_it': -1}
    iter_times = []
    for string in lis:
        m = pattern.match(string)
        if m is None: continue
        if 's/it' not in m.group(): continue

        if verbose: print(m.group())
        # expect m.group()== 80%|████████  | 4/5 [03:58<00:59, 59.31s/it]
        t = m.group().split('[')[1].split(']')[0].split(',')[1].strip(' ')
        if t[-4:]!='s/it': continue
        t = t[:-4]
        iter_times.append(float(t))

    print('-'*verbose)
    if iter_times: ret['tqdm_it'] = int(np.median(iter_times))
    return ret


parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='cora', help='{cora, pubmed, citeseer}.')
parser.add_argument('--model', type=str, default='GCN', help='{SGC, DeepGCN, DeepGAT}')

if __name__ == '__main__':
    lis = ['80%|████████  | 4/5 [03:58<00:59, 59.31s/it]Epoch: 005, Train Loss: 0.303, Test RMSEN loss: 7.868/0.16787654',
           ' 0%|          | 0/5 [00:00<?, ?it/s]Epoch: 001, Train Loss: 0.467, Test RMSEN loss: 19.223/0.16787654',
            ' 20%|██        | 1/5 [01:02<04:11, 62.88s/it]Epoch: 002, Train Loss: 0.449, Test RMSEN loss: 18.814/0.16787654']
    print(filter_tqdm(lis))
    exit()
    lis = ['train: 11.73926s\n', 'abc','test: 1.1s\n', 'test: 1.2s']
    print(filter_logtime(lis))
    exit()
    slicestr()

    exit()
    s = "There are 2 apples for 4 persons"
    s_list = ['ap', 'person']
    print(hasany(s, s_list))

    exit()
    args = parser.parse_args()
    print(arg2str(args))
    exit()
    # Driver code
    s = "CuFeMV"
    print(camel_case_split(s))

    exit()
    s = 'bace  bbbp  bbp  chembl_filtered  clintox  esol  freesolv  hiv  lipophilicity  mutag  muv  pcba  ptc_mr  sider  tox21  toxcast  unsupervised  zinc_standard_agent'
    shell2script(s)

    exit()
    lis = ['test_pretrain_epochs_501_eph_18_test_epochs_50', 'test_pretrain_epochs_501_eph_201_test_epochs_50',
           'test_pretrain_epochs_501_eph_20_test_epochs_50', 'test_pretrain_epochs_501_eph_199_test_epochs_50',
           'test_pretrain_epochs_501_eph_22_test_epochs_50', 'test_pretrain_epochs_501_eph_2000_test_epochs_50']
    field_vals = sort_model_name(lis, field='eph', delimiter='_').field_vals
    print(field_vals)

    sorted_files = sort_model_name(lis, field='eph', delimiter='_').sort_by_field()
    pprint(sorted_files)
