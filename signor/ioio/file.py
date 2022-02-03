# Created at 2021-02-14
# Summary: info about a file
import pathlib
import datetime

from signor.format.format import red, banner


def ftime(path):
    # modification time of a file
    # https://bit.ly/380fxXI

    # time of last access of contents (atime), time of last modification of contents (mtime), and time of last modification of the inode (metadata, ctime)
    fname = pathlib.Path(path)
    assert fname.exists(), f'No such file: {fname}'  # check that the file exists
    stat = fname.stat() # os.stat_result(st_mode=33204, st_ino=26889027, st_dev=2050, st_nlink=1, st_uid=1001, st_gid=1002, st_size=25140, st_atime=1613337944, st_mtime=1613335159, st_ctime=1613335159)
    mtime = stat.st_mtime
    # mtime = datetime.datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S')
    mtime = datetime.datetime.fromtimestamp(mtime).strftime('%m-%d %H:%M')
    return mtime

import subprocess
def lastkline(path, k=1):
    # todo: somehow the output of tail -n 3 path is different from file.readlines for tqdm output
    ret = []
    with open(path) as file:
        lines = file.readlines()[-k:]
        for i, line in enumerate(lines):
            if not line.startswith('Epoch'): continue
            # if line == '\n': continue
            # if line.startswith('train') and (not line.startswith('train: 100')): continue
            ret.append(str(i).zfill(3) + '  ' + line)

    if ret and 'Error' in ret[-1]:
        ret = [red(l) for l in ret]
    return ''.join(ret)

def nol(path):
    # number of line
    num_lines = sum(1 for _ in open(path))
    return num_lines

    # with open(path) as f:
    #     text = f.readlines()
    #     print(text)
    #     size = len(text)
    # return size


if __name__ == '__main__':
    path = '/home/chen/fromosu/Signor/signor/parallel_args/test_deepmind/d_3_dataset_Spring_delta_t_0.005_dev_cpu_lr_1e-3_mlp_num_hidden_layers_2_mode_rollout_n_4_n_epoch_500_n_systems_100_num_message_passing_steps_5_traj_len_100.log'
    # path = '/home/chen/fromosu/Signor/signor/./parallel_args/tfn-perm/bllevel_1_data_version_1_dev_cuda_loss_RMSEN_n_epoch_50_n_graph_300_prop_perm_seed_1.log'
    # path = '/home/cai.507/Documents/DeepLearning/Signor/signor/./parallel_args/gnn/bllevel_1_data_version_1_depth_3_device_cuda:1_loss_RMSEN_n_epoch_100_n_graph_300_prop_perm_seed_1.log'
    print(ftime(path))
    print(nol(path))
    print(lastkline(path, k=40))