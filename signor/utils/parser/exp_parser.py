# Created at 5/2/21
# Summary: experiment parser; used for monitor experiments with different hyperparameters; modified
#          modifed from graph/physics/e3nn/perm_playground/parser.py

from time import time

from signor.ioio.file import ftime

t0 = time()

import os.path as osp
from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from signor.configs.util import dict2name, dict_product, load_configs, name2dict
from signor.format.format import banner, red, pf, full_print
from signor.ioio.dir import sig_dir
from signor.monitor.time import curtime
from signor.utils.dict import merge_two_dicts
from signor.utils.parser.metric import get_metric
print(red(f'Import takes {time()-t0}'))
# exit()

def get_field(s, field):
    ret = s.split(f'{field}')[-1]
    return float(ret)


class Parser(object):
    def __init__(self, dir):
        self.dir = dir
        # self.keys = ['n_layers', 'n_train', 'seed']
        self.lastk = 10
        self.cnt = 0

    def set_keys(self, keys):
        self.keys = keys

    def set_lastk(self, lastk):
        self.lastk = lastk

    def default_dict(self):
        print(red('Using defalut dict'))
        return {}  # dict(zip(self.keys, self.defaultvals))

    def set_vars_range(self, vars_range, verbose=True):
        """
        # vars = ['n_epoch', 'n_graph', 'loss', 'bllevel', 'seed', 'device', 'prop']
        # args_range = [[100],
        #               [300],
        #               ['RMSEN'],
        #               [1],
        #               [1, 2, 3, 4, 5],
        #               ['cuda:1'],
        #               ['form']]
        # vars_range = dict(zip(vars, args_range))
        :param vars_range:
        :param verbose:
        :return:
        """
        assert isinstance(vars_range, dict)
        vars_range.pop('file', None)
        vars_range.pop('python', None)

        self.vars_range = vars_range
        self.keys = []
        for k, v in self.vars_range.items():
            if len(v) > 1:
                self.keys.append(k)
        self.defaultvals = [-1] * len(self.keys)
        if verbose:
            pprint(vars_range)

    def from_yaml(self, f):
        # read a file from config
        configs = load_configs(f)
        return configs

    def get_content(self, file, verbose=False):
        if file is None: file = self.random_file()
        if verbose:
            print(f'{self.cnt}: {file}', )
        self.cnt += 1

        with open(file) as f:
            content = f.readlines()
        return content

    def get_metric_hist(self, file=None, plot=False):
        """
        # todo: this is not really used anywhere
        :param file:
        :return:    epoch  train_loss  test_cart_loss  trivial_loss
                0  291.0       0.205           0.463      0.165745
                1  292.0       0.205           0.394      0.165745
                2  293.0       0.206           0.272      0.165745
                ....
        """
        content = self.get_content(file, verbose=True)
        df = get_metric(content, lastk=self.lastk, end=None, aggreator=None)
        if plot:
            try:
                print(file)
                self.cnt += 1
                df.plot(x='epoch', logy=False, title=file)
                plt.show()
            except AttributeError:
                pass
        return df

    def get_ret_per_file(self, file=None, verbose=False):
        """
        from a file, get result and put into an df
        :param file: like bllevel_1_loss_RMSEN_n_epoch_50_n_graph_300_seed_1.log
        :param verbose: if true, print perf_dict
        :return: a dict like {train_loss: 10, test_loss: 4, ...}
        """
        raise NotImplementedError

    def pprint(self):
        # pretty print vars-range
        for k, v in self.vars_range.items():
            if len(v) == 1:
                print(f'{k}: {v[0]}')
        print(f'Now: {curtime()}')
        print()

    def get_perf(self, verbose=True):
        files = self.all_logs()
        perf = []
        error_cnt = 0

        for file in tqdm(files):
            try:
                perf.append(self.get_ret_per_file(file, verbose=True))
            except (ValueError):
                error_cnt += 1
                print(f"{red('ParseError')}({error_cnt}) for {file}")
            except FileNotFoundError:
                error_cnt += 1
                print(f"{red('FileNotFoundError')}({error_cnt}) for {file}")
        df = pd.DataFrame(perf)

        if verbose:
            import pandas
            pandas.set_option('display.max_rows', 100)
            banner(f'Performance Table')
            self.pprint()
            full_print(df)
            banner()
        return df

    def random_file(self):
        """ get a random log file """
        d = {k: v[0] for k, v in self.vars_range.items()}
        f = dict2name(d)
        f = osp.join(self.dir, f + '.log')
        print(red(f'Random pick a file \n{f}'))
        return f

    def all_logs(self):
        cfc_list = dict_product(self.vars_range, shuffle=False)
        files = []
        for d in cfc_list:
            f = dict2name(d)
            f = osp.join(self.dir, f + '.log')
            files.append(f)
        return files


class SubParser(Parser):
    def __init__(self, dir):
        super().__init__(dir)

    def _parse_line(self, line):
        """
        'Epoch: 256, Train Loss: 0.11663, Test loss: 0.11687\n' or
        'Epoch: 497, Train Loss: 0.70544, Test loss: 0.70069, Train Time: 9.6, Test time: 4.8\n'
        :param line:
        :return:
        """
        line = line.rstrip('\n')
        entries = line.split(',')
        train_loss = entries[1].split('Train Loss: ')[-1]
        test_loss = entries[2].split('Test loss: ')[-1]
        train_time = -1 if len(entries) <= 3 else entries[3].split('Train Time: ')[-1]
        test_time = -1 if len(entries) <= 3 else entries[4].split('Test time: ')[-1]
        ret = train_loss, test_loss, train_time, test_time
        ret = list(map(float, ret))
        return ret

    def _parse_lines(self, lines):
        print(red(f'Average over last {min(len(lines), self.lastk)}'))
        if 'Error' in lines[-1]:
            print(red(lines[-1]))
        ret = [self._parse_line(line) for line in lines]
        ret = np.array(ret)
        ret = np.mean(ret, axis=0)
        ret[2], ret[3] = int(ret[2]), int(ret[3])
        return ret.tolist()

    def get_ret_per_file(self, file=None, verbose=False):
        """
        from a file, get result and put into an df
        :param file: like bllevel_1_loss_RMSEN_n_epoch_50_n_graph_300_seed_1.log
        :param verbose: if true, print perf_dict
        :return: a dict like {train_loss: 10, test_loss: 4, ...}
        """
        d = name2dict(file, list(self.keys))

        ret = {'n_epoch': -1, 'train_loss': -1, 'test_loss': -1, 'time': ftime(file)}
        content = self.get_content(file, verbose=verbose)
        content = [line for line in content if line.startswith('Epoch')]

        if content:
            ret['n_epoch'] = len(content)
            ret['train_loss'], ret['test_loss'], ret['train_time'], ret['test_time'] = \
                self._parse_lines(content[-self.lastk:])
        return merge_two_dicts(d, ret)


if __name__ == '__main__':
    from signor.graph.physics.learning_to_simulate.torch_model.configs.run import DIR

    f = osp.join(DIR, 'toy_example', 'n_epoch.yaml') # 'gn_kw.yaml'
    print(red(f))
    vars_range = load_configs(f)
    parallel_argsf = vars_range['file'][0].split('/')[-1][:-3]

    dir = osp.join(sig_dir(), 'parallel_args', f'{parallel_argsf}')
    P = SubParser(dir)
    P.set_vars_range(vars_range)
    P.set_lastk(10)

    # f = '/home/chen/fromosu/Signor/signor/parallel_args/test_deepmind/d_3_dataset_Spring_delta_t_0.005_dev_cpu_lr_1e-3_mlp_num_hidden_layers_4_mode_one_step_n_8_n_epoch_500_n_systems_100_num_message_passing_steps_2_traj_len_100.log'
    # P.get_ret_per_file(file=f, verbose=True)
    df = P.get_perf(verbose=True)
    # exit()

    vals = ['train_loss']# ['n_epoch', 'train_loss',] # []  #
    for v in vals:
        banner(v)
        table = pd.pivot_table(df, values=v, index=P.keys[0],
                               columns=P.keys[1:],
                               aggfunc=[lambda x: f'{pf(np.mean(x), 4)}±{pf(np.std(x), 4)}({len(x)})', ])
        print(table)
