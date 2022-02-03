""" implement loop over args quickly
    python file --arg 1
    python file --arg 2
    python file --arg 3

    run the above cmds simulatenously and save the result at
    ./parallel_args/file/1.log
    ./parallel_args/file/2.log
    ./parallel_args/file/3.log
    ...
"""

# todo: add the runnng time of every file
# todo: handle true/false arg
# todo: handle multiple argsp
import argparse
import os
import random
from pprint import pprint

from signor.configs.util import dict2arg, dict_product, dict2name, load_configs, load_cfg, load_short_configs
from signor.format.format import one_liner, banner
from signor.ioio.dir import make_dir, sig_dir, cur_dir, fdir
from signor.monitor.time import tf
from signor.utils.cli import runcmd
from signor.utils.list import isflatlist
from signor.utils.shell.cmd_ import keep_cmd
from signor.monitor.time import curtime_
from signor.configs.gpu import n_gpu, random_cuda_dev_prefix
import os.path as osp
from signor.ioio.dir import simplifyPath
class parallel_args():
    def __init__(self, python, file, arg, arg_range):
        """
        :param python:
        :param file: **.py include non-modified args
        :param arg: modified arg. a list of args
        :param arg_range: a list of lists

        Note fromyaml is similar to __init__
        """
        self.python = python
        if '-u' not in self.python:
            self.python += ' -u '
        if file[-1] != ' ': file = file + ' '
        self.file = file
        print(self.file)

        self.arg = [arg] if isinstance(arg, str) else arg
        self.range = [arg_range] if isflatlist(arg_range) else arg_range
        assert len(self.arg) == len(self.range), \
            f'arg and range are of different length. arg: {self.arg}. range: {self.range}'

        self.dir = f'{sig_dir()}parallel_args/{self._getfile(self.file)}/'
        make_dir(self.dir)

    def fromyaml(self, f='configs/equiv.yaml'):
        f = os.path.join(eval(cur_dir()), f)
        try: # long configs
            configs = load_configs(f)
        except TypeError: # short configs, such as /home/chen/fromosu/Signor/signor/graph/IGN/invariantgraphnetworks-pytorch/main_scripts/config/yamltest.yaml
            configs = load_short_configs(f)
        assert set(['python', 'file']).issubset(set(configs.keys()))

        self.python = configs['python'][0] + ' -u'

        # parse file directory
        if configs['file'][0][0]=='/': # absolute path
            self.file = configs['file'][0]
        elif configs['file'][0][0]=='.': # relative path
            print(f)
            self.file = osp.join(fdir(f), configs['file'][0])
            print(f'relative path: {self.file}')
        else:
            raise NotImplementedError

        # simplify file path
        self.file = simplifyPath(self.file)

        keys = set(configs.keys()) - set(['python', 'file'])
        self.arg = list(keys)
        self.range = [configs[k] for k in self.arg]
        self.dir = f'{sig_dir()}parallel_args/{self._getfile(self.file)}/'
        make_dir(self.dir)

    def _process_single_cmd(self, **kwargs):
        """ extracted from gen_cmds """
        pass

    def _get_logf(self, d, flat=True):
        """ generate log file from arg dict like {x: 1, y: 2}
        """
        if len(d) == 1:
            return f'{self.dir + str(list(d.values())[0])}.log'

        if flat:
            f = dict2name(d) + '.log'
        else:
            f = dict2name(d, flat=False) + 'log'
        f = os.path.join(self.dir, f)
        return f

    def gen_cmds(self, **kwargs):
        cmds = []

        cfg_dict = dict()
        for i in range(len(self.arg)):
            arg, arg_range = self.arg[i], self.range[i]
            cfg_dict[arg] = arg_range
        cfc_list = dict_product(cfg_dict, shuffle=False)  # a list of dict

        for d in cfc_list:
            arg = dict2arg(d)
            cmd = ' '.join([self.python, self.file, arg])
            log_f = self._get_logf(d)  # dict2name(d) + '.log'

            if kwargs.get('overwrite', False) or keep_cmd(f'{cmd} >> {log_f}'):  # overwrite
                os.system(f'echo {cmd} > {log_f}')
            else:
                continue

            if kwargs.get('direct', False) or kwargs.get('nohup', False):
                cmd += f' >> {log_f}'  # append

            if kwargs.get('nohup', False):
                cmd = 'nohup ' + cmd + ' &'

            cmds.append(cmd)
        self.cmds = cmds

    @tf
    def run(self, print_only=True, gnu=False, **kwargs):
        """
        :param print_only:
        :param gnu: use gnu parallel
        :param kwargs:
        :return:
        """
        self.gen_cmds(**kwargs)

        if gnu:  # gnu parallel
            cmds = [cmd.replace('nohup', '') for cmd in self.cmds]
            cmds = [cmd.replace('&', '') for cmd in cmds]
            exe = not print_only
            write_cmds_for_parallel(cmds, exe=exe, nohup=True, jobs=kwargs.get('jobs', 5),
                                    setprefix=kwargs.get('setprefix', ''),
                                    random_gpu=kwargs.get('random_gpu', False))
            return

        for cmd in self.cmds:
            runcmd(cmd, print_only=print_only)

    def _getfile(self, file):
        """ from "abc.py --a 10" get abc """
        res = file.split(' ')[0]
        if 'CUDA_VISIBLE_DEVICES' in res:
            res = file.split(' ')[1]
        assert '.py' in res, f'support .py file only. Got {res}/{file}'
        res = res.split('/')[-1]
        return res[:-3]


def write_cmds_for_parallel(cmds, exe=False, nohup=False, shuffle=False, **kwargs):
    """
    write a list of cmds into a script that will be used for gnu parallel bin
    :param cmds: a list of cmds
    :param exe: execute with parallel

    kwargs
    ---
    :param multi_gpu: if true, use multiple gpus
    :param random_gpu: assign random gpu
    :param ignore_prev: whether to ignore previous results
    :param file: specify file name. Use it to run two sets of commands concurrently

    Example:
    cmds = ['sleep 10'] * 20
    write_cmds_for_parallel(cmds, exe=True, jobs=10)

    """
    assert isinstance(cmds, list)
    cmds = [one_liner(cmd) for cmd in cmds]
    cmds = [cmd + ' 2>&1' for cmd in cmds]  # direct both output and error to file
    if shuffle: random.shuffle(cmds)

    # set prefix here
    prefix0 = ' ' if kwargs.get('multi_gpu', None) in [None, False] else f'CUDA_VISIBLE_DEVICES=0 '
    prefix1 = ' ' if kwargs.get('multi_gpu', None) in [None, False] else f'CUDA_VISIBLE_DEVICES=1 '
    if 'setprefix' in kwargs:
        prefix0 = prefix1 = kwargs.get('setprefix', '')

    cmds = [cmd + '\n' for cmd in cmds if cmd[-1] != '\n']

    new_cmds = []
    for i, cmd in enumerate(cmds):
        prefix = prefix0 if i % 5 in [0, 1] else prefix1
        if kwargs.get('random_gpu', False):
            cmd = random_cuda_dev_prefix() + cmd
        else:
            cmd = prefix + cmd
        new_cmds.append(cmd)
    cmds = new_cmds

    file = kwargs.get('file', f'tmp_{curtime_()}')
    file = f'{sig_dir()}utils/scheduler/{file}.sh'
    with open(file, 'w') as f:
        f.writelines('#!/usr/bin/env bash\n')
        f.writelines(cmds)

    banner(f'parallel executing cmds from {file}')
    os.system(f'head -n 2 {file} | tail -1')
    cmd = f' time parallel --progress --jobs {kwargs.get("jobs", 5)} < {file} '

    if exe:
        if nohup: cmd = 'nohup ' + cmd  # + '&'
        if kwargs.get('background', False): cmd += ' &'
        banner(cmd)
        os.system(cmd)
    else:
        banner(f'No Exe ({len(new_cmds)} cmds): ' + cmd)
    return file


parser = argparse.ArgumentParser(description='PyTorch implementation of pre-training of graph neural networks')
parser.add_argument('--pretrain_model', type=str, default='neph_50_geph_60', help='')
parser.add_argument('--xt_idA', type=int, default=46744)

if __name__ == '__main__':
    # python
    args = parser.parse_args()

    python = 'python'
    xt_idA = args.xt_idA
    pretrain_model = args.pretrain_model  # 'neph_50_geph_60'

    # file = f'graph/cgcnn/code/chem/finetune.py --xt_id 3402 ' + \
    #        f'--device 1                      --epochs 200 ' + \
    #        f'--model_file                   {xt_model_dir()}mp-ids-{xt_idA}/band_gap/chem_graph_pretrain/{pretrain_model}.pth ' + \
    #        f'--pretrain_model               {pretrain_model}.pth ' + \
    #        f'--seed                         0 ' + \
    #        f'--transfer ' + \
    #        f'--xt_idA                       {xt_idA} '

    file = 'graph/physics/e3nn/perm_playground/tfn-perm.py --n_epoch 30 '
    # 'graph/permeability/model/gnn.py'
    # 'graph/permeability/model/gin.py --n_epoch 200 --dev cpu --coarse '
    # 'graph/hea/baseline.py --emb one_hot --bm --mat elastic_both --norm_w'  #
    # 'graph/hea/paper/pred_result.py'
    # 'graph/hea/baseline.py --emb one_hot  --mat nonequal_ec '
    # 'graph/hea/better_emb.py'
    # 'graph/hea/inverse_pred.py '
    # 'graph/sparsifier/model.py' \
    # ' # # 'configs/scikit_hyper.py' # 'graph/cgcnn/run_cgcnn.py' #  # 'graph/hea/util.py' # 'graph/hea/baseline.py' #

    arg = \
        ['n_graph', 'loss', 'bllevel', 'seed']
    # ['seed', 'n_train', 'n_layers']
    # ['idx', 'loss', 'run', 'prop']
    # ['prop', 'method']
    # ['prop', 'method', 'exclude']
    # 'idx'
    # "name"
    # 'idx'  # 'slice'  # 'sample' # 'emb' #
    # 'gnn_type'

    arg_range = \
        [[50, 100, 300],
         ['RMSEN', ],
         [1],
         [1, 2, 3, 4, 5]
         # [0,1,2,3,4],
         # [10, 20,30,40,50],
         # [2,3,4,5]
         # [0], #[-1,0,1,2],
         # ['l1_ratio'],
         # [1,2,3,4,5],
         # ['bulk']
         # ['c11', 'c12', 'c44'],
         # ['bulk', 'equil_volume', 'lattice_constant'], # [ 'equil_volume', 'lattice_constant', 'bulk', 'energy'],
         # ['bulk', 'equil_volume', 'lattice_constant', 'B_prime', 'c11', 'c12', 'c44', 'young_modulus', 'G_vrh',
         #  'poisson_ratio', 'B/G', 'A_u'],
         # ['B_G'],
         #  ['knn', 'mlp', 'rf', 'gbt', 'svm', 'linear_reg', 'mlp_skorch', 'deepset'],
         # ['svm', 'mlp', 'gbt'],
         # exclude_list(sample_ratio=1) # ['-1', '0', '1', '2', '3', '4', '5', '6', '7', '8']
         ]  # [1000, 3000] #['final_energy_per_atom', 'formation_energy_per_atom']# xt_props # ['final_energy_per_atom', 'formation_energy_per_atom', 'efermi']# ['elasticity.K_VRH', 'elasticity.G_VRH', 'elasticity.poisson_ratio'] #   ['one_hot', 'neural', 'naive'] #[0,1,2,3,4,5,6] #
    # list(range(11))
    # ['emb1', 'emb2', 'emb3', 'emb4', 'emb5', 'emb6', 'emb7', 'emb8']

    # list(range(108))
    # list(range(6))
    # ['mlp_skorch', 'deepset']
    # ['gin', 'gcn', 'graphsage', 'gat']

    prun = parallel_args(python, file, arg, arg_range)
    # prun.fromyaml(f='configs/gnn.yaml')
    # prun.run(nohup=True, print_only=False, gnu=True, jobs=20, )
    # exit()

    for file in [
        # 'gnn-lr'
        # 'cnn-benchmark'
        # 'equivariant-cnn-ucsd',
        #  'gnn-ucsd',
        # 'gnn-aug',
        # 'cnn-depth',
        #  'cnn-ucsd',
        # 'gnn-pool'
        # 'depth-gnn-v1',
        # 'gnn-feat-engineer'
        #  'equiv-gnn-ucsd',
        # 'equiv-gnn-depth',
        # 'equiv-gnn-tune',
        # 'gnn-compare-data-version'
        # 'gnn-mask',
        'gnn-conv',
    ]:  # ['depth-gnn']:
        prun.fromyaml(f=f'configs/{file}.yaml')
        prun.run(nohup=True, print_only=False, gnu=True, jobs=6, overwrite=False)  # setprefix='CUDA_VISIBLE_DEVICES=1')
    exit()

    prun.run()
    banner()

    prun.run(nohup=True)
    banner()

    prun.run(direct=True)
