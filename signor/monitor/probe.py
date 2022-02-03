""" get the shape of everything """

from time import time

from signor.monitor.time import timefunc

t0 = time()
import collections
import inspect
import sys
from functools import partial
from warnings import warn

import numpy as np
import pandas as pd
import scipy.sparse
import torch

from signor.format.format import banner, red, pf
from signor.monitor.stats import stats

from signor.utils.nan import nan1
from signor.utils.np import tonp

t1 = time()
print(f'import summary takes {pf(t1 - t0, 2)}s')


# import torch_geometric # important: move torch_geometric into functions since import pyg is slow

# @profile
# def fun():
#     return 1
# fun()


def summary(x, name='x', terminate=False,
            skip=False, delimiter=None, precision=3,
            exit=False, highlight=False, sizecap=False):
    if highlight:
        name = red(name)

    if skip:
        print('', end='')
        return ''

    if x is None:
        print(f'{name} is None')

    if isinstance(x, list):
        print(f'{name}: a list of length {len(x)}')
        l = len(x)
        x = x[:2]
        for i, _x in enumerate(x):
            summary(_x, f'{i}th element in list')
        if l > 2: print('...')

    elif isinstance(x, scipy.sparse.csc.csc_matrix):
        min_, max_ = x.min(), x.max()
        mean_ = x.mean()

        std1 = np.std(tonp(x))
        x_copy = x.copy()
        x_copy.data **= 2
        std2 = x_copy.mean() - (x.mean() ** 2)  # todo: std1 and std2 are different
        pf_ = partial(pf, precision=precision)
        mean_, min_, max_, std1, std2 = list(map(pf_, [mean_, min_, max_, std1, std2]))

        line0 = '{:>10}: csc_matrix ({}) of shape {:>8}'.format(name, str(x.dtype), str(x.shape))
        line0 = line0 + ' ' * max(5, (45 - len(line0)))
        # line0 += 'Nan ratio: {:>8}.'.format(nan_ratio(x_))
        line1 = '  {:>8}(mean) {:>8}(min) {:>8}(max) {:>8}(std1) {:>8}(std2) {:>8}(unique) ' \
            .format(mean_, min_, max_, std1, std2, -1)
        line = line0 + line1
        print(line)

    elif isinstance(x, (np.ndarray,)):
        if x.size > 232960 * 10 and sizecap:
            print(f'Sizecap: skip {name}')
            return
        x_ = tonp(x)
        ave_ = np.mean(x_)
        median_ = np.median(x_)
        max_ = np.max(x_)
        min_ = np.min(x_)
        std_ = np.std(x_)
        unique_ = len(np.unique(x_))
        pf_ = partial(pf, precision=precision)
        ave_, min_, max_, median_, std_, unique_ = list(map(pf_, [ave_, min_, max_, median_, std_, unique_]))

        line0 = '{:25}'.format(name) + '{:15}'.format(str(x.dtype)) + '{:15}'.format(str(x.shape))
        # line0 = '{}  {}  {:>3}'.format(name, str(x.dtype), str(x.shape))
        # line0 += '\n' # line0 + ' ' * max(5, (45 - len(line0)))
        # line0 += 'Nan ratio: {:>5}.'.format(nan_ratio(x_))
        line1 = '{:>8}(mean) {:>8}(min) ' \
                '{:>8}(max) {:>8}(median) ' \
                '{:>8}(std) {:>8}(unique)'.format(ave_, min_, max_, median_, std_, unique_)
        line = line0 + line1

        # commented out temporarily for learning to simulate project
        # if np2set(x_) <= set([-1, 0, 1]):
        #     ratio1 = np.sum(x_ == 1) / float(x_.size)
        #     ratio0 = np.sum(x_ == 0) / float(x_.size)
        #     line +=  '|| {:>8}(1 ratio)'.format(pf(ratio1, 3)) # + '|| {:>8}(1 ratio) {:>8}(0 ratio)'.format(pf(ratio1, 3), pf(ratio0, 3))

        if nan1 in x_:
            nan_cnt = np.sum(x_ == nan1)
            line += f'nan_cnt {nan_cnt}'

        # f'{name}: array of shape {x.shape}.'
        print(line)
        # print(f'{name}: a np.array of shape {x.shape}. nan ratio: {nan_ratio(x)}. ' + line)

    elif isinstance(x, (torch.Tensor)):
        if x.numel() > 232965 * 10 and sizecap:
            print(f'Sizecap: skip {name}')
            return
        if x.ndim == 0:
            print(f'{name}: zero length tensor. {x.item()}')
        else:
            x_ = tonp(x)
            ave_ = np.mean(x_)
            median_ = np.median(x_)
            max_ = np.max(x_)
            min_ = np.min(x_)
            std_ = np.std(x_)
            unique_ = len(np.unique(x_))

            pf_ = partial(pf, precision=2)
            ave_, min_, max_, median_, std_, unique_ = list(map(pf_, [ave_, min_, max_, median_, std_, unique_]))
            line = '{:>8}(mean) {:>8}(min) ' \
                   '{:>8}(max) {:>8}(median) ' \
                   '{:>8}(std) {:>8}(unique)'.format(ave_, min_, max_, median_, std_, unique_)

            print('{:25}'.format(name) + \
                  '{:15}'.format(str(x.data.type())[6:].rstrip('ensor')) + \
                  '{:15}'.format(str(x.size())[11:-1]) + line)
        # print(line)
        # print(f'{name}: a Tensor ({x.data.type()}) of shape {x.size()}')

    elif isinstance(x, tuple):
        print(f'{name}: a tuple of shape {len(x)}')
        if len(x) < 6:
            for i, ele in enumerate(x):
                summary(ele, name=f'\t{i}th ')

    elif isinstance(x, (dict, collections.defaultdict)):
        # print('-'*10 + f'{name}: dict of len {len(x)}' + '-'*10)
        banner(f'{name}: dict of len {len(x)}', compact=True, ch='-', length=70)
        n_key = 0
        for k, v in x.items():
            summary(v, name=k, sizecap=sizecap)
            n_key += 1
            if n_key > 4:
                print('...')
                return

    elif isinstance(x, pd.DataFrame):
        from collections import OrderedDict

        dataType_dict = OrderedDict(x.dtypes)
        banner(text=f'start summarize a df ({name}) of shape {x.shape}', ch='-')
        print('df info')
        print(x.info())
        print('\n')

        print('head of df:')
        # print(tabulate(x, headers='firstrow'))
        print(x.head())
        print('\n')

        try:
            print('continuous feats of Dataframe:')
            cont_x = x.describe().T
            print(cont_x)
            print(cont_x.shape)
            print('\n')
        except ValueError:
            print('x.describe().T raise ValueError')

        try:
            print('non-cont\' feats (object type) of Dataframe:')
            non_cont = x.describe(include=[object]).T
            print(non_cont)
            print(non_cont.shape)
        except ValueError:
            print('x.describe(include=[object]).T raise ValueError')

        banner(text=f'finish summarize a df ({name}) of shape {x.shape}', ch='-')

    elif isinstance(x, (int, float)):
        print(f'{name}(float): {x}')

    elif isinstance(x, str):
        print(f'{name}(str): {x}')

    elif summary_pygeo(x, name=name):
        pass
    else:
        print(f'{x}: \t\t {type(x)}')
        if terminate:
            exit(f'NotImplementedError for input {type(x)}')
        else:
            pass

    if delimiter is not None:
        assert isinstance(delimiter, str)
        print(delimiter)

    if exit:
        sys.exit()


def summary_pygeo(data, stat=False, precision=2, name=None):
    try:
        import torch_geometric
    except ImportError:
        raise Exception('Check pytorch geometric install.')

    if not isinstance(data, torch_geometric.data.data.Data):
        return False

    print(f'Summary of {name} (pyG.Data):')
    for k, v in data:
        print('     ', sep=' ', end=' ')
        if isinstance(v, torch.Tensor):
            if v.ndim == 1:
                summary(v, name=k, precision=precision)
            else:
                if v.size()[1] != 0:
                    summary(v, name=k, precision=precision)
                else:
                    warn(f'Empty edge index: {v}')
        elif isinstance(v, str):
            summary(v, k)
        else:
            NotImplementedError

    if stat:
        for k, v in data:
            stats(v, var_name=k)

    return True


def dummy_pygeo_data():
    import torch_geometric
    x = np.random.random((100, 3))
    y = np.random.random((10,))
    return torch_geometric.data.data.Data(x=x, y=y)


def nan_ratio(x):
    """ http://bit.ly/2PL7yaP
    """
    assert isinstance(x, np.ndarray)
    try:
        return np.count_nonzero(np.isnan(x)) / x.size
    except TypeError:
        return '-1 (TypeError)'


def dump_args(func, concise=True):
    """Decorator to print function call details - parameters names and effective values.
        https://bit.ly/2w3B44Q
    """

    def wrapper(*args, **kwargs):
        fname = func.__qualname__
        func_args = inspect.signature(func).bind(*args, **kwargs).arguments  # OrderedDict
        banner(f'{func.__module__}.{red(fname)} ')

        print(red(f'{fname} input:'))
        if concise:
            for k, v in func_args.items():
                summary(v, name=str(k), sizecap=False)
        else:
            func_args_str = ', '.join('{} = {!r}'.format(*item) for item in func_args.items())
            print(f'{func.__module__}.{fname} ( {func_args_str} )')

        out = func(*args, **kwargs)
        banner(red(f'{fname} output:'))
        if concise:
            print(f'{summary(out, name=f"Return value of {fname}", sizecap=False)}\n')
        else:
            print(out)
        return out

    return wrapper


concise_dump_args = partial(dump_args, concise=True)


@concise_dump_args
def test(a, b=4, c='blah-blah', *args, **kwargs):
    return a


if __name__ == '__main__':
    exit()
    x = np.random.rand(100, 100, 100)
    summary(x, 'asuperlongnamevariable')
    exit()
    summary(torch.tensor(x), 'asuperlongnamevariable')

    test(1, 2, 'abc', 4, 5, d=6, g=12.9)
    exit()
    x = ['1', 'abc', 'xyz']
    summary(np.array(x))
    exit()
    test(1)
    test(1, 3)
    test(1, d=5)
    x = test(np.random.random((3, 3)))
    summary(x)

    exit()
    x = np.array([[1, 2, np.nan], [np.nan] * 3])
    print(nan_ratio(x))
    exit()

    x = np.random.random((100, 3))
    summary(x, name='x')
    summary(x, name='x', skip=True)
    print('-' * 10)
    exit()

    data = dummy_pygeo_data()
    summary_pygeo(data)

    exit()
    x = np.random.random((100, 3))
    summary(x, name='x')
    x = (x, x, x)
    summary(x)

    if hasattr(x, '__str__'):
        if len(x.__str__()) > 100:
            print(f'x is of shape {x.shape}')
