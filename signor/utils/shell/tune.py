""" sweep over multiple args
    can be used together with parallel_args. (sweep over one parameter in parallel)

"""
from time import sleep

from signor.configs.util import dict_product, dict2name, dict2arg
import os
import warnings
from signor.configs.gpu import gpu_monitor

from signor.format.format import banner


class tune():
    def __init__(self, cmd, args, ranges):
        """

        :param cmd:
        :param args: [--a, --b, ...]
        :param ranges: [[1,2,3], [4,5,6], ...]
        """
        assert isinstance(cmd, str)
        assert isinstance(args, list)
        assert isinstance(ranges, list)
        assert len(args) == len(ranges)

        params = dict()
        for i in range(len(args)):
            k, v = args[i], ranges[i]
            params[k] = v

        self.cmd = cmd # base command
        warnings.warn(f'{self.cmd} should not be run at background.')

        self.params = params
        self.all_args = dict_product(self.params, shuffle=False)

    def run_cmds(self, run=False, save = False):
        """ generate commands
        :param save: if True, save result to a file
        """
        # all_args = dict_product(self.params) # each element is one particular combination
        for d in self.all_args:
            arg = dict2arg(d)
            cmd = self.cmd + f' {arg}'
            if save:
                name = dict2name(d)
                # todo: setup dir as well
                cmd += f' > {name}'
            banner(f'start to run {cmd}')

            if run:
                os.system(cmd)
            # sleep(20*60)
            # gpu_monitor().query_ava(self, device=1, t=50, mem_lim=8000) # todo: modify this according to application



if __name__ == '__main__':
    base_script = 'python ./utils/shell/parallel_args.py'
    model_list = ['neph_0_geph_0'] #['neph_0_geph_0', 'neph_0_geph_30', 'neph_0_geph_60', 'neph_100_geph_60', 'neph_150_geph_90', 'neph_150_geph_60', 'neph_0_geph_90', 'neph_50_geph_0', 'neph_100_geph_0', 'neph_150_geph_0']
    t = tune(base_script, ['pretrain_model', 'xt_idA'], [model_list, [3402, 27430, 46744]])
    t.run_cmds(run=False, save=True)