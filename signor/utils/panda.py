""" used to further process the result generated from set_target.py. """
import time
t0 = time.time()
import sys
from copy import deepcopy
import numpy as np

import pandas as pd
import torch
t1 = time.time()
print(t1-t0)

from signor.format.format import banner, pf
from signor.ioio.dir import netproj_dir
from signor.ioio.dir import signor_dir
from signor.ioio.mem import f_size
from signor.monitor.probe import summary
from signor.monitor.time import timefunc
from signor.utils.df import norm_df, range_lookup
from signor.utils.np import np2set
t2 = time.time()
print(t2-t0)


def sparsity_filter(df, thres=.95):
    """ filter a 0/1 data frame by the sparsity of columns
        http://bit.ly/2HYAqrt
    """
    assert np2set(df.to_numpy()) <= set([0, 1])
    df_ = df.loc[:, (df == 0).mean() < thres]
    print(f'Before/after filter the column number is {df.shape[1]}/{df_.shape[1]}')
    return df_

def meanstd_tuple_agg(nums, n_sig=2):
    # given a series of numbers, compute its mean and std
    valid_nums, test_nums = [], []
    for v, t in nums:
        valid_nums.append(v)
        test_nums.append(t)
    return pf(np.mean(valid_nums), 3), len(nums), f'{pf(np.mean(test_nums), n_sig)}$\pm${pf(np.std(test_nums), n_sig)}'


def meanstd_agg(nums, n_sig=1):
    # given a series of numbers, compute its mean and std
    mean = pf(np.mean(nums), n_sig)
    std = pf(np.std(nums), n_sig)
    n = len(nums)
    if std != 0:
        return f'{mean}±{std} ({n})'
    else:
        return f'{mean}'

def tuplize(df, columns=[], newcol='tuple_col'):
    df[newcol] = df[columns].apply(tuple, axis=1)
    print(df)
    return df

d = {'task_ids': [['mp-1', 'mp-2'], ['mp-3', 'mp-4'], ['mp-8']],
     'feat1': ['a', 'b', 'c'],
     'feat2': [7, 8, 9]
     }  # http://bit.ly/2VmQmf4 test mp-ids filter


class df_query():
    def __init__(self, pretrain=True, bucket=True, verbose=False, normdf = True):
        """
        used to further process the result generated from set_target.py.
        :param pretrain:
        :param verbose:
        :param regression: if true, use regression feature, instead of one hot feat.

        For future reference:
                                      count          mean           std        min           25%           50%           75%            max

        band_gap                   123788.0      1.080988      1.527087   0.000000      0.000000      0.120950      1.909800      17.902300
        elasticity.G_VRH           123788.0      0.151133      0.473712   0.000000      0.000000      0.000000      0.000000       3.724522
        elasticity.K_VRH           123788.0      0.186824      0.574731   0.000000      0.000000      0.000000      0.000000       2.759668
        final_energy_per_atom      123788.0     -5.895754      1.759795 -14.331771     -7.049743     -6.038707     -4.794120      -0.016100
        elasticity.poisson_ratio   123788.0      0.029000      0.092699  -1.130000      0.000000      0.000000      0.000000       0.600000
        formation_energy_per_atom  123788.0     -1.417737      1.208961  -4.618361     -2.411896     -1.451662     -0.465557       5.355523
        efermi                     123788.0      2.902779      2.877250 -14.201352      0.892102      2.847844      4.862703      19.275775

        """
        self.dir = f'{netproj_dir()}data/raw_data/df/'
        self.pretrain = pretrain
        self.logprops = ['elasticity.G_VRH', 'elasticity.K_VRH']
        self.bucket = bucket
        self.id_props = ['task_ids', 'task_id']  # todo: more generic
        self.rawdf = self.load_df()
        # range_lookup(self.df)
        # self.df = self.rawdf

        if normdf:
            saveto = f'{signor_dir()}graph/cgcnn/data_/material-data/norm_bucket_{self.bucket}_pretrain_{self.pretrain}.csv'
            self.df = norm_df(deepcopy(self.rawdf), cols=self.rawdf.columns, saveto=saveto)  # normalize y
        else:
            self.df = self.rawdf

        if verbose:
            summary(self.df, self.filename)

        self.bad_cnt = 0
        self.cnt = 0
        self.dummy = 0  # used as feat  when mp-id is not found
        self.task_ids2row()
        banner(f'Done df_query init with pretrain {pretrain}, bucket {bucket}')

    def _nalog10(self, x):
        from signor.utils.nan import nan1
        if x <= 0:
            return nan1
        else:
            return np.log10(x)

    def range_lookup(self, **kwargs):
        range_lookup(self.rawdf, **kwargs)
        # summary(self.rawdf)
        range_lookup(self.df, **kwargs)

    def load_df(self):
        """ load raw df """
        f = self.dir + f'/bucket_{self.bucket}_pretrain_{self.pretrain}.csv'
        self.filename = f
        f_size(f)
        df = pd.read_csv(f)

        if self.bucket == False:
            # important: do not filter too early
            # important(updated): not discard any data
            logprops = [prop for prop in self.logprops if prop in df.columns]
            for p in logprops:
                # df = df[df[p] >= 0]
                df[p] = df[p].apply(self._nalog10)
            df = df.reset_index()
        return df

    def list_cols(self, df):
        lis = list(df.columns)
        for i, name in enumerate(lis):
            print(i, name)

    @timefunc
    def query_row(self, key='task_id', val=['mp-1']):
        """ from task_id, get the corresponding feat
            implemented first but not used often.
        """

        assert key == 'task_id'
        res = self.df[self.df[key].isin(val)] # return df
        print(res.shape)
        banner('finish query')

        if len(res) == 0 and len(val) == 1:
            self.bad_cnt += 1
            print(f'{self.bad_cnt}/{self.cnt}: {val[0]} not found in {key} in self.df({len(self.df)}) for pretrain={self.pretrain}')

        assert len(res) == len(val), f'Length mismatch. ' \
            f'Query vals is of length {len(val)} while result is {len(res)}. ' \
            f'The diff is {self.set_dff(set(list(res[key])), set(val))}'
        return res

    def get_feat(self, key='task_ids', val=['mp-1'], verbose=False, slice=None):
        """
        get the feature of one row and then convert to a tensor
        :param key: task_id/task_ids
        :param val: a list of mp-ids
        :param verbose:
        :param slice: by default, return all all go_target_downstream, else return a subset of list(range(7))
        :return:
        """
        self.cnt += 1
        assert key in ['task_id', 'task_ids']
        if slice is not None: assert self.pretrain == False # make sure slice only used for go_target_downstream

        if key == 'task_id':
            df = self.query_row(key=key, val=val)
        else:
            indices = [self.task_ids2row_dict.get(val_, -1) for val_ in val]

            if -1 in indices:  # if didn't find task_ids just use feat of the 1st row
                # tmp = self.df['task_ids']
                # self.task_ids = []
                # for row in tmp:
                #    self.task_ids += self.str2lis(row)
                # self.task_ids = dict(zip(self.task_ids, range(len(self.task_ids))))
                # self.task_ids['mp-1059837']
                # self.task_ids['mvc-989527']

                print(f'{self.bad_cnt}/{self.cnt}: {val} not found in {key} in self.df({len(self.df)}) for pretrain={self.pretrain}. Indices is {indices}')
                self.bad_cnt += 1
                indices = [1 if x == -1 else int(x) for x in indices]
            df = self.df.iloc[indices]

        df = df.drop(columns=self.id_props)
        if 'index' in df.columns: df = df.drop(columns=['index'])

        if verbose:
            summary(self.df, 'self.df')
            summary(df, 'df')

        res = df.to_numpy()
        if 'bucket_True' in self.filename:
            assert np2set(res) <= set([0, 1]), f'feat2set is {np2set(res)}'
            if slice is not None: res = res[slice]
            return torch.tensor(res).type(torch.LongTensor)
        if len(val) == 1:
            res = res.reshape((res.shape[1],))
            if slice is not None: res = res[slice]
            return torch.tensor(res)

    def task_ids2row(self):
        """ compute a dict where key is task_ids and val is correspoinding row number
            called once at init since it's quite expansive.
        """
        assert 'task_ids' in list(self.df.columns)
        idx_ids_dict = dict(self.df['task_ids'])

        for k in idx_ids_dict.keys():
            idx_ids_dict[k] = self.str2lis(idx_ids_dict[k])

        # reverse key, val in idx_ids_dict
        ids_idx_dict = dict()
        for k, v in idx_ids_dict.items():
            for _k in v:
                ids_idx_dict[_k] = k

        self.task_ids2row_dict = ids_idx_dict

    def set_dff(self, s1, s2):
        """ set difference """
        return set(s1) - set(s2), set(s2) - set(s1)

    def query_col(self):
        pass

    def str2lis(self, s):
        # convert "[\\'mp-850232\\', \\'mp-892915\\', \\'mp-904406\\']" into a list of str
        s = s.replace('[', '')
        s = s.replace(']', '')
        s = s.replace('\'', '')
        s = s.replace('\\', '')
        s = s.replace(' ', '')
        s = s.split(',')
        return s

def filter_df(df, row, vals):
    """ given a df, leave rows whose value is in vals """
    # todo
    df = df[df[row]]
    return df


import argparse
parser = argparse.ArgumentParser(description='test')
parser.add_argument('--bucket', action='store_true')

if __name__ == '__main__':
    df = pd.DataFrame(d)
    tuplize(df, columns=['feat1', 'feat2'], newcol='new_feat')


    exit()
    args = parser.parse_args()
    vals = ['mp-1']  # ['mp-1', 'mp-100']

    q = df_query(pretrain=False, bucket=args.bucket)
    tsr = q.get_feat(key='task_ids', val=vals, verbose=True, slice=list(range(3)))
    print(tsr)
    summary(tsr, 'tsr_from_task_ids')
    # summary(q.df.to_numpy(), 'q.df')

    # q.range_lookup(subplots = True, figsize=(8, 8))
    sys.exit()


    q = df_query(pretrain=False)
    tsr = q.get_feat(val=vals)
    summary(tsr, 'tsr')

    exit()

    q = df_query(pretrain=False)
    q.query_row(val=['mp-1', 'mp-100'])
    exit()
    df = pd.DataFrame(d)
    df = df.explode('task_ids')
    print(df.loc[df['task_ids'] == 'mp-1'])
    # assert all query mp-ids is a subset of task_ids

    print(df.loc[df['task_ids'].isin(['mp-1', 'mp-3', 'mp-9'])])  # todo: how to handle mp-9
    exit()
    key_arr = df['task_ids'].to_numpy()
    val_arr = df.drop(['task_ids'], axis=1).to_numpy()

    assert len(key_arr) == val_arr.shape[0]
    assert val_arr.ndim == 2
    df_dict = dict()
    for i in range(len(key_arr)):
        k, v = key_arr[i], val_arr[i]
        df_dict[k] = v
    print(df_dict)

    exit()
    f, t = np.bool_(False), np.bool_(True)
    d = {'col1': [f, t], 'col2': [f, t]}
    df = pd.DataFrame(d)
    # df.astype({'col1': object})
    df['col1'] = df['col1'].map({True: [True], False: [False]})
    print(df)
    print(df.dtypes)
