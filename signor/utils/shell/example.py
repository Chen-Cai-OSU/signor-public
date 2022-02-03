# Created at 6/7/21
# Summary: example for han fu
import os
import argparse
import os.path as osp

from signor.ioio.dir import cur_dir
from signor.utils.shell.parallel_args import parallel_args

parser = argparse.ArgumentParser(description='test')
parser.add_argument('--file', type=str, default='rollout_opt', help='')
parser.add_argument('--overwrite', action='store_true')
parser.add_argument('--print_only', action='store_true')
parser.add_argument('--jobs', type=int, default=1)

if __name__ == '__main__':
    args = parser.parse_args()
    python = 'python'
    file = 'x/y.py'  # 'graph/physics/e3nn/perm_playground/tfn-perm.py --n_epoch 30 '
    arg = [None]
    arg_range = [[None]]

    prun = parallel_args(python, file, arg, arg_range)

    ################################# equivariance_test #################################
    dir = eval(cur_dir())
    prun.fromyaml(f=osp.join(dir, f'example.yaml'))
    prun.run(nohup=True, print_only=args.print_only, gnu=True, jobs=args.jobs, overwrite=args.overwrite)
