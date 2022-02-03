import os
from pprint import pprint

import yaml

from signor.configs.util import dict2arg
from signor.configs.util import dict_product


def load_hyperparams(method='RF'):
    """
    load yaml config file for different mode
    :param method: umap
    :return: a dict of form {'n_neighbors': [10, 15, 20], 'min_dist': [0.1, 1], 'metric': ['euclidean', 'hamming'], 'n_components': [2]}
    """

    dir = '/home/cai.507/Documents/DeepLearning/material/Wei/baseline/hyperparameter/'
    dir = '/Users/admin/Documents/osu/Research/Signor/signor/viz/kwargs/'
    file = method + '.yaml'
    with open(dir + file, 'r', encoding='utf-8') as ymlfile:
        cfg = yaml.load(ymlfile, yaml.FullLoader)
    cfg_ = {}
    for k, v in cfg.items():
        cfg_[k] = v['values']
    return cfg_


import argparse

parser = argparse.ArgumentParser(description='Baseline')
parser.add_argument('--print', action='store_true', help='print only')
parser.add_argument('--code', action='store_true', help='use code')
parser.add_argument('--no_direct', action='store_true', help='do not write to a file')
parser.add_argument('--data_', default='mp', help='graph data_ to load', choices=['george', 'mp', 'tianxie'])

parser.add_argument('--target', default='elasticity.K_VRH', type=str, help='')
parser.add_argument('--device', default='0', type=str, help='')


class cmd_issuer():

    def __init__(self, python, file, out_dir=None, config_file=None, out_file='', device=1):
        self.python = python
        self.file = file
        self.out_dir = out_dir
        self.config_file = config_file
        self.device = f'CUDA_VISIBLE_DEVICE={device}'
        self.out_file = out_file

        self.cmds = None

    def load_configs(self):
        """ load configs from yaml file and convert to a dict """
        with open(self.config_file, 'r', encoding='utf-8') as ymlfile:
            cfg = yaml.load(ymlfile, yaml.FullLoader)
        cfg_dic = {}
        for k, v in cfg.items():
            cfg_dic[k] = v['values']
        return cfg_dic

    @staticmethod
    def fileExist(file, skip=True):
        """ used for check if skip the cmd if the output file has been written alredy """
        if os.path.isfile(file):
            with open(file) as f:
                cont = f.readlines()
                if len(cont) > 300 and skip:
                    print(f'{file} exists. Skip.')
                    return True
                else:
                    return False
        else:
            return False

    def gen_cmds(self, shuffle=False, direct=False):
        cfg_dic = self.load_configs()
        cfgs = dict_product(cfg_dic, shuffle=shuffle)  # a list of dicts
        cmds = []

        for cfg in cfgs:
            arg = dict2arg(cfg)
            cmd = ' '.join([self.device, self.python, self.file, arg])
            outfile = self.out_dir + self.out_file
            if self.fileExist(outfile, skip=False):
                continue

            if direct: cmd += f' > {outfile} &'
            cmds.append(cmd)
        self.cmds = cmds
        return cmds

    def exe_cmds(self, print_only=False):
        assert self.cmds is not None
        for cmd in self.cmds:
            print(cmd + '\n')

        if not print_only:
            for cmd in self.cmds:
                pprint(cmd)
                os.system(cmd)
                print('-' * 50 + '\n')


if __name__ == '__main__':
    args = parser.parse_args()
    target = args.target
    data = args.data

    python = 'nohup /home/cai.507/anaconda3/bin/python -u '
    gcn_file = '/home/cai.507/Documents/DeepLearning/material/Wei/baseline/GCN.py'
    cgcnn_file = '/home/cai.507/Documents/DeepLearning/material/code/main.py'
    out_dir = '/home/cai.507/Documents/DeepLearning/material/Wei/baseline/log/'
    config_file = '/Users/admin/Documents/osu/Research/Signor/signor/viz/kwargs/tsne.yaml'

    cmd_issuer = cmd_issuer(python, gcn_file, out_dir, config_file=config_file)
    cmd_issuer.load_configs()
    cmd_issuer.gen_cmds()
    cmd_issuer.exe_cmds(print_only=True)
