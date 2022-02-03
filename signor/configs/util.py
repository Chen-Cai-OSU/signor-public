import random
import sys
from collections import OrderedDict, defaultdict
from copy import deepcopy
from itertools import product

import yaml

from signor.format.format import red
from signor.ioio.dir import sig_dir

random.seed(42)


def dict_product(d, shuffle=False):
    """
    :param d: d = {'x': [1,2,3], 'y':['a', 'b', 'c']}
    :param shuffle:
    :return: a list of dict

    >>> d = {'x': [1, 2, 3], 'y': ['a', 'b', 'c']}
    >>> res = dict_product(d)
    >>> assert len(res) == 9

    """

    assert isinstance(d, dict)
    keys = d.keys()
    results = []

    for element in product(*d.values()):
        results.append(dict(zip(keys, element)))

    if shuffle: random.shuffle(results)
    print(f'There are {red(len(results))} combinations.')
    return results


def my_function(a, b):
    """
    >>> my_function(2, 3)
    6
    >>> my_function('a', 3)
    'aaa'
    """
    return a * b


def dict2arg(d, verbose=0, sacred=False):
    """
    :param d: {'n_epoch': 300, 'bs': 32, 'n_data': 10, 'scheduler': True}
    :param sacred: if true. with sacred arg version. with n_epoch=300 bs=32
    :return: --scheduler --n_epoch 300 --bs 32 --n_data 1000

    >>> d =  {'n_epoch': 300, 'bs': 32, 'n_data': 10, 'scheduler': True}
    >>> assert dict2arg(d) == '--scheduler  --bs 32 --n_data 10 --n_epoch 300'
    """

    d = OrderedDict(sorted(d.items(), reverse=False))
    copy_d = deepcopy(d)
    if verbose: print(d)
    arg = 'with ' if sacred else ''

    for k, v in OrderedDict(d).items():
        if v is True:
            arg = arg + f"'{k}={v}' " if sacred else arg + f'--{str(k)} '
            copy_d.pop(k)
        elif v is False:  # do nothing
            arg = arg + f"'{k}={v}' " if sacred else arg
            copy_d.pop(k)
        else:
            pass

    copy_d = OrderedDict(copy_d)
    for k, v in copy_d.items():
        if sacred:
            arg += f"'{k}={v}' "
        else:
            arg += f' --{k} {v}'

    if verbose:
        print(arg)
        print('-' * 10)

    return arg


def dict2name(d, flat=True, ignore_device=True, remove_underscore=False):
    """
    :param d: {'n_epoch': 300, 'bs': 32, 'n_data': 10, 'scheduler': True}
    :param remove_underscore: remove the underscore in keys
    :return: bs_32_n_data_10_n_epoch_300_scheduler_True if flat else
    return   bs_32/n_data_10/...
    """
    if ignore_device:
        for k, v in d.items():
            if isinstance(v, str) and 'cuda' in v:
                d[k] = 'cuda'

    assert isinstance(d, dict)
    keys = list(d.keys())
    keys.sort()
    name = ''

    if flat:
        for k in keys:
            newk = rm_underscore(k) if remove_underscore else k
            newv = rm_underscore(d[k]) if remove_underscore else d[k]
            name += f'{newk}_{newv}_'
        return name[:-1]
    else:
        for k in keys:
            newk = rm_underscore(k) if remove_underscore else k
            newv = rm_underscore(d[k]) if remove_underscore else d[k]
            name += f'{newk}_{newv}/'
        return name


def rm_underscore(s):
    s = str(s)
    return ''.join(s.split('_'))


def _extraV(s, k):
    # todo: still needs improvement
    # todo: add documentation
    # s = '/home/chen/fromosu/Signor/signor/parallel_args/test_deepmind/d_3_dataset_Spring_delta_t_0.005_dev_cpu_lr_1e-3_mlp_num_hidden_layers_2_mode_one_step_n_4_n_epoch_500_n_systems_100_num_message_passing_steps_2_traj_len_100.log'
    assert k in s, f'{k} not in {s}'
    tmp = s.split(f'{k}_')
    if len(tmp) == 2:
        return tmp[1].split('_')[0]
    elif tmp[2].startswith('later_'):  # ugly hack
        return tmp[1].split('_')[0]
    elif tmp[1].startswith('viz2rec') or tmp[1].startswith('bl1') or tmp[1].startswith('bl2') or tmp[1].startswith(
            'deep_sim'):  # another ugly hack
        return tmp[1].split('_')[0]
    else:
        import re
        pattern = re.compile(f"{k}_\d+")
        try:
            start, end = pattern.search(s).span()
            return s[start:end].split('_')[1]
        except AttributeError:
            return tmp[2][:6]


def name2dict(fname, keys, verbose=False):
    """
    :param fname: file like bllevel_1_loss_RMSEN_n_epoch_50_n_graph_300_seed_1.log # n_layers_2_n_train_10_seed_0.log
    :param keys: a list of keys of interest
    :param strkeys: a list of keys whose value should be str
    :return: a dict
    """
    assert isinstance(keys, list)
    assert len(keys) == len(set(keys))
    if '.log' in fname:
        fname = fname.split('/')[-1].rstrip('.log')
    else:
        raise NotImplementedError(f'{fname} does not end with log')

    keys, vals = keys, []  # need to set manually
    for k in keys:
        v = _extraV(fname, k)
        vals.append(v)
        # fname = fname.lstrip(v + '_')
        # print(fname)

    ret = dict(zip(keys, vals))
    if verbose: print(ret)
    return floatify_dict(ret)


def floatify_dict(d):
    # convert values into float when ever possible
    newd = {}
    for k, v in d.items():
        if v in ['False', 'True']:
            newd[k] = v
        elif v in [True, False]:
            newd[k] = v
        else:
            try:
                newd[k] = float(v)
            except ValueError:
                newd[k] = v
    return newd


def val_from_name(name, key):
    """ from name a_1_b_2 get the value corresponding key """
    # todo: only works when key doesn't contain _
    assert '_' in name
    name = name.split('_')
    assert key in name, f'key {key} not in {name}'
    key_idx = name.index(key)
    val_idx = key_idx + 1
    return float(name[val_idx])


class subset_dict():
    " get the subset of origianl dict"

    def __init__(self, d):
        assert isinstance(d, dict)
        self.d = d
        self.keys = d.keys()

    def include_args(self, keys):
        assert isinstance(keys, list)
        ret = []
        for key in keys:
            try:
                ret.append(self.d[key])
            except KeyError:
                exit(f'{key} in not the key of {self.d}')
        return tuple(ret)

    def include(self, keys):
        assert isinstance(keys, list)
        _d = dict()
        for key in keys:
            try:
                _d[key] = self.d[key]
            except KeyError:
                exit(f'{key} in not the key of {self.d}')
        return _d

    def exclude(self, keys):
        assert isinstance(keys, list)
        _d = deepcopy(self.d)
        for key in self.keys:
            if key in keys:
                _d.pop(key)
        return _d


def test_load_configs():
    # https: // bit.ly / 3nk2v2y
    import os
    from pprint import pprint
    cur_dir = os.path.dirname(os.path.realpath(__file__))
    f = os.path.join(cur_dir, 'example2.yaml')
    with open(f, 'r', encoding='utf-8') as ymlfile:
        cfg = yaml.load(ymlfile, yaml.FullLoader)
    pprint(cfg)


def load_cfg(f):
    with open(f, 'r', encoding='utf-8') as ymlfile:
        cfg = yaml.load(ymlfile, yaml.FullLoader)
    return cfg

def load_short_configs(f):
    cfg = load_cfg(f)
    for k, v in cfg.items():
        if not isinstance(v, list):
            cfg[k] = [v]
    return cfg


def load_configs(f):
    """ load configs from yaml file and convert to a dict
        Can be used for hyperparameter search.
    """
    cfg = load_cfg(f)
    assert len(cfg) > 0, f'Empty dict from {f}'
    cfg_dic = defaultdict(int)

    for k, v in cfg.items():
        # if there is only use one gpu, find the ununsed one
        if k in ['device', 'dev'] and len(v['values']) == 1 and 'cuda' in v['values'][0]:
            from signor.configs.gpu import free_gpu
            cfg_dic[k] = [f'cuda:{free_gpu()}']
        else:
            cfg_dic[k] = v['values']

    if 'train_indices' in cfg_dic and cfg_dic['train_indices'][0][:2] == '"\\':
        cfg_dic['train_indices'][0] = cfg_dic['train_indices'][0][2:-2]
        cfg_dic['test_indices'][0] = cfg_dic['test_indices'][0][2:-2]

    return cfg_dic


def load_deepset_yaml(id=0):
    import torch.nn as nn
    f = f'{sig_dir()}ml/classic/configs/deepset.yaml'
    cfg_dic = load_configs(f)

    cfg_dic['extractor_nl'] = [nn.Identity]  # [nn.ReLU, nn.ELU, nn.Identity] #
    cfg_dic['regressor_nl'] = [nn.ReLU, nn.ELU]

    del cfg_dic['device']
    del cfg_dic['verbose']

    # remark: used this to quickly test an idea
    # cfg_dic['extractor_nl'] = [nn.ELU]# [nn.ReLU, nn.ELU] #
    # cfg_dic['in_features'] = [94]
    # cfg_dic['set_features'] = [10, 30, 50]
    # cfg_dic['regressor_hidden_dim'] = [[100, 50]]# [[10], [30], [50]]

    # good for 4_non_equal
    cfg_dic['regressor_nl'] = [nn.ReLU, nn.ELU]
    cfg_dic['extractor_nl'] = [nn.Identity, nn.ELU, nn.ReLU]
    # cfg_dic['extractor_hidden_dim'] = [[41]]  # [[94], [94, 50], [50, 50]] # todo: add back
    cfg_dic['extractor_hidden_dim'] = [[94]]  # [[94], [94, 50], [50, 50]]
    # cfg_dic['in_features'] = [41] # todo: add back
    cfg_dic['in_features'] = [94]
    cfg_dic['bn'] = [True, False]
    # cfg_dic['set_features'] = [41, ]
    cfg_dic['set_features'] = [94, ]  # todo: add back

    model_params = dict_product(cfg_dic, shuffle=True)
    assert id < len(model_params), f'{id} larger than num of combinations({len(model_params)})'
    opt_param = {'batch_size': 32, 'lr': 0.001, 'verbose': 0, 'max_epochs': 300}
    return model_params[id], opt_param


if __name__ == '__main__':
    load_deepset_yaml()
    exit()
    test_load_configs()
    exit()
    s = 'd_3_dataset_Spring_delta_t_0.005_dev_cuda_egnn_n_hidden_64_egnn_n_layers_4_equiv_True_lr_1e-3_mode_one_step_n_8_n_epoch_100000_n_systems_100_proj_vn_proj_later_False_seed_1_traj_len_100_vn_n_hidden_64_vn_n_layers_4'
    print(_extraV(s, 'proj'))
    sys.exit()
    import pprint

    s = '/home/chen/fromosu/Signor/signor/parallel_args/test_deepmind/d_3_dataset_Spring_delta_t_0.005_dev_cpu_lr_1e-3_mlp_num_hidden_layers_2_mode_one_step_n_4_n_epoch_500_n_systems_100_num_message_passing_steps_2_traj_len_100.log'
    print(_extraV(s, 'n'))
    sys.exit()
    fname = 'bllevel_1_data_version_0_device_cuda_emb_dim_50_loss_RMSEN_n_copy_2_n_epoch_500_n_graph_300_prop_perm_seed_1_test_aug_True_train_aug_True.log'
    name2dict(fname, ['data', 'aug', 'device'], verbose=True)
    sys.exit()

    f = os.path.join(sig_dir(), 'utils', 'shell', 'configs', 'cnn-benchmark.yaml')
    print(load_configs(f))

    sys.exit()
    d = {'a': 'abc', 'b': -1, 'cd': 0.3, 'device': 'cuda:3'}
    print(dict2name(d))
    sys.exit()
    d = {'a': 'abc', 'b': -1, 'cd': 0.3}
    print(floatify_dict(d))

    sys.exit()
    fname = 'n_layers_2_n_train_10_seed_0.log'
    keys = ['n_layers', 'seed', 'n_train']
    name2dict(fname, keys, verbose=True)
    sys.exit()
    d = {'x': 1, 'y': 2}

    for f in [True, False]:
        name = dict2name(d, flat=f)
        print(name)

    sys.exit()
    all = load_deepset_yaml()
    summary(all, 'all')
    banner()
    pprint(all)

    sys.exit()

    d = {'x': [1, 2, 3], 'y': ['a', 'b', 'c']}
    res = dict_product(d, shuffle=True)
    pprint(res)

    sys.exit()
    name = 'ac_1_b_2'
    print(val_from_name(name, 'a'))

    sys.exit()

    print(sorted(d.items(), reverse=False))
    print(dict2arg(d))
    pass

    d = {'n_epoch': 300, 'bs': 32, 'n_data': 10, 'scheduler': True}
    print(dict2name(d))

    print(subset_dict(d).include(['n_epoch', 'bs']))
    print(subset_dict(d).exclude(['n_epoch', 'bs']))
