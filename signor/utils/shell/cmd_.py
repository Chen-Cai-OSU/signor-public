# Created at 2021-02-14
# Summary: cmd related helper functions

import os
import os.path as osp

from signor.format.format import red
from signor.ioio.dir import sig_dir
from signor.ioio.file import ftime, nol
from signor.ioio.keyboard import yes_or_no


def logfile(cmd):
    # find the log file
    cmdsplit = cmd.split('>>')
    assert len(cmdsplit) == 2, f'Number of >> in {cmd} is not 1!'
    f = cmdsplit[-1].strip()
    f = f.split(' ')[0]
    ret = osp.join(sig_dir(), f)
    return ret


def keep_cmd(cmd):
    path = logfile(cmd)
    if not os.path.isfile(path):
        return cmd

    mtime = ftime(path)
    n_lines = nol(path)
    q = f'{"-" * 100}\n' \
        f'Going to run: {cmd}\n\n' \
        f'Keep file {red(path)} or not?\n' \
        f'Modification time: {mtime}. Size: {n_lines} loc.\n' \
        f''
    if yes_or_no(question=q):
        return None
    else:
        return cmd


if __name__ == '__main__':
    cmd = ' ~/anaconda3/envs/signor/bin/python -u graph/permeability/model/gnn.py --bllevel 1 --data_version 1 --depth 3 --device cuda:1 --loss RMSEN --n_epoch 100 --n_graph 300 --prop perm --seed 1 >> ./parallel_args/gnn/bllevel_1_data_version_1_depth_3_device_cuda:1_loss_RMSEN_n_epoch_100_n_graph_300_prop_perm_seed_1.log  2>&1'
    logfile(cmd)
    keep_cmd(cmd)
