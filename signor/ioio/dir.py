import json
import os
import os.path as osp
import subprocess
import time
import warnings
from os import listdir
from os.path import isfile, join
from pprint import pprint

import torch

from signor.format.format import pf
from signor.ioio.mem import getmem
from signor.utils.system import detect_sys
from os.path import expanduser


def log_dir():
    return f'{sig_dir()}../data/log'

def tb_dir():
    dir = f'{home()}/Documents/DeepLearning/Signor/data/tensorboard/'
    make_dir(dir)
    return dir


def assert_ucsd_pc():
    import socket
    hostname = socket.gethostname()
    assert hostname == 'u112222', hostname


def make_file():
    pass


def make_dir(dir):
    # has side effect

    if dir == None:
        return

    if not os.path.exists(dir):
        os.makedirs(dir)


def assert_dir_exist(dir, warn=False):
    try:
        assert os.path.isdir(dir)
    except AssertionError:
        msg = f'Directory {dir} does not exist'
        if warn:
            warnings.warn(msg)
        else:
            raise Exception(msg)


def assert_file_exist(f):
    assert os.path.isfile(f), f'File {f} does not exist'


def all_dirs(dir):
    """ list all directoreis under dir """
    ret = next(os.walk(dir))[1]  # http://bit.ly/39ZBOGq
    return sorted(ret)


def save_obj(obj, dir, file):
    t0 = time.time()
    make_dir(dir)
    with open(dir + file, 'wb') as f:
        pickle.dump(obj, f)
        print(f'Saved objects Takes {int(time.time() - t0)}.')

    cmd = f'du -sh {dir + file}'
    print(cmd)
    os.system(cmd)


def load_obj(dir, file):
    t0 = time.time()
    with open(dir + file, 'rb') as f:
        data = pickle.load(f)
        print('Load file %s. Takes %s s' % (os.path.join(dir, file), int(time.time() - t0)))
    return data


def tmp_dir():
    return '/tmp/'


def mktemp(verbose=1, tmp_dir=True):
    r""" create a tmp file or tmp dir
    https://stackoverflow.com/questions/3503879/assign-output-of-os-system-to-a-variable-and-prevent-it-from-being-displayed-on
    :param tmp_dir: return dir if true. return file (full path) if false
    """

    cmd = 'mktemp'
    tmp_f = subprocess.check_output(cmd,
                                    stderr=subprocess.STDOUT)  # bytes string like b'/var/folders/wf/0h72413x7yg9smz6sbw0_h9c0000gp/T/tmp.13Y13kAp\n'
    tmp_f = str(tmp_f, 'utf-8')[:-1]  # http://bit.ly/3b3zBLj

    if verbose:
        print(f'create a tmp dir at {tmp_f}')

    if tmp_dir:
        cmd = f'rm {tmp_f}'
        os.system(cmd)

        # tmp_f = '/var/folders/wf/0h72413x7yg9smz6sbw0_h9c0000gp/T/tmp.IJbJsQGm'
        _tmp = ['/'] + tmp_f.split('/')[1:-1]  # ['/', 'var', 'folders', 'wf', '0h72413x7yg9smz6sbw0_h9c0000gp', 'T']
        tmp_dir = os.path.join(*_tmp)
        return tmp_dir
    else:
        return tmp_f


class torch_io():
    def __init__(self, dir=None, verbose=False):
        if dir is None:
            self.dir = self.tmp_dir()
        else:
            self.dir = dir
        self.verbose = verbose

    def tmp_dir(self):
        # https://unix.stackexchange.com/questions/174817/finding-the-correct-tmp-dir-on-multiple-platforms
        return mktemp(verbose=1, tmp_dir=True)

    def dump(self, data, filename):
        from signor.monitor.probe import summary
        summary(data, name='data_')

        f = os.path.join(self.dir, filename)
        torch.save(data, f)

        if self.verbose:
            print(f'dump data_ of {pf(getmem(data), 1)}M at {f} with success')
        del data

    def load(self, filename):
        f = os.path.join(self.dir, filename)
        data = torch.load(f)

        if self.verbose:
            print(f'load data_ of {pf(getmem(data), 1)}M at {f} with success')

        return data


class io():
    def __init__(self, dir, file, saver='json'):
        self.dir = dir
        self.file = file
        self.obj = None
        self.saver = saver

    def save_obj(self, obj):
        self.obj = obj
        t0 = time.time()
        if not os.path.exists(self.dir): make_dir(self.dir)

        if self.saver == 'pickle':
            with open(self.dir + self.file, 'wb') as f:
                pickle.dump(self.obj, f)
        elif self.saver == 'json':
            with open(self.dir + self.file, 'wb') as f:
                json.dump(self.obj, f)
        else:
            print('Saver is not specified. ')
            raise IOError
        print('Saved obj at %s. Takes %s' % (self.dir + self.file, time.time() - t0))

    def load_obj(self):
        t0 = time.time()
        try:
            if self.saver == 'pickle':
                with open(self.dir + self.file, 'rb') as f:
                    res = pickle.load(f)
            elif self.saver == 'json':
                with open(self.dir + self.file, 'r') as f:
                    res = json.load(f)
            else:
                print('Saver is not specified. ')
                raise IOError

            print('Load file %s. Takes %s' % (str(self.file), time.time() - t0))
            return res
        except (IOError, EOFError) as e:
            print('file %s does not exist when loading with %s' % (self.dir + self.file, self.saver))
            return 0

    def rm_obj(self):
        if os.path.exists(self.dir + self.file):
            os.remove(self.dir + self.file)
            print('Delete file %s with success' % self.file)
        else:
            print('file %s does not exist' % self.file)


def allf(dir, name_only=False, sort=False, verbose=False):
    # get all files in a directory #todo: add pattern matcher
    if verbose:
        print(listdir(dir))

    if name_only:
        onlyfiles = [f for f in listdir(dir) if isfile(join(dir, f))]
    else:
        onlyfiles = [join(dir, f) for f in listdir(dir) if isfile(join(dir, f))]
    onlyfiles.sort()
    return onlyfiles


def dl_dir(osu=False):
    if osu:
        return f'{home()}/Documents/DeepLearning/'
    else:
        return f'/data/chen/cai.507/cai.507/Documents/DeepLearning/'  # ucsd


def fdir(f):
    # get the directory of a file belongs to
    return '/'.join(f.split('/')[:-1])


def home(archive=False):
    # https://bit.ly/2ZPFE2R
    if archive:
        assert_ucsd_pc()
        return '/data/chen/cai.507/cai.507/'
    else:
        return expanduser("~")


def hea_sys_dir():
    return os.path.join(home(), 'Dropbox', 'Wei_Data', 'HEA_System', '')  # '/home/cai.507/Dropbox/Wei_Data/HEA_System/'


def pretrain_dir():
    return f'{home()}/Documents/DeepLearning/pretrain-gnns/'


def home_dir():
    from pathlib import Path
    home = str(Path.home())
    return home


def denali_dir():
    return os.path.join(sig_dir(), 'viz', 'denali', 'hea', '')


def hea_emb_dir():
    return os.path.join(sig_dir(), '..', 'data', 'hea_emb', '')


def physics_dir():
    return os.path.join(sig_dir(), 'graph', 'physics', '')


def sig_dir():
    hdir = home_dir()
    if detect_sys() == 'Linux':
        if 'cai.507' in hdir:
            return '/home/cai.507/Documents/DeepLearning/Signor/signor/'
        elif 'chen' in hdir:
            return '/home/chen/fromosu/Signor/signor/'
        elif '/home/ubuntu' in hdir:  # amazon ec2
            return '/home/ubuntu/proj/Signor/signor/'
        else:
            raise NotImplementedError
    elif detect_sys() == 'Darwin':
        if 'hanfu' in hdir:
            return '/Users/hanfu/Downloads/Signor/signor/'
        elif 'checai' in hdir:
            return '/Users/checai/Documents/amazon/config/signor/signor/'
        else:
            return '/Users/admin/Documents/osu/Research/Signor/signor/'
    else:
        NotImplementedError


def config_dir():
    return osp.join(signor_dir(), 'utils', 'shell', )


def signor_dir():
    return sig_dir()
    # dir = f'{home()}/Documents/DeepLearning/Signor/signor/'


def get_dir(f):
    import os
    f = f'{home()}/Documents/DeepLearning/Signor/signor/graph/cgcnn/code/run_finetune.py'
    os.path.dirname(f)


def netproj_dir():
    return f'{home()}/Dropbox/2020_Spring/Network/proj/'


def xt_model_dir():
    return f'{home()}/Dropbox/2020_Spring/Network/proj/data/TianXie/model/'


def xt_cif_dir():
    dir = f'{home()}/Dropbox/2020_Spring/Network/proj/data/TianXie/cif/'
    if not os.path.exists(dir):
        dir = f'{home(archive=True)}/Dropbox/2020_Spring/Network/proj/data/TianXie/cif/'
        assert os.path.exists(dir)
    return dir


def mat_dir():
    dir = dl_dir()
    return os.path.join(dir, 'material', '')


def curdir(str=True):
    return osp.dirname(osp.realpath(__file__))


def cif_sample_dir():
    dir = curdir()
    return os.path.join(dir, '..', 'graph', 'cgcnn', 'data_', 'sample-regression', '')


def find_files(dir='./', suffix='.txt', verbose=False, include_dir=False):
    # find all files in a dir ends with say .txt
    # https://stackoverflow.com/questions/3964681/find-all-files-in-a-directory-with-extension-txt-in-python
    assert dir[-1] == '/'
    files = []
    for file in os.listdir(dir):
        if file.endswith(suffix):
            if include_dir: file = os.path.join(dir, file)
            if verbose: print(file)
            files.append(file)
    return files


def rm_files(dir='./', suffix='.txt', verbose=True):
    files = find_files(dir=dir, suffix=suffix, include_dir=True)
    rmfiles(files, verbose=False)
    if verbose:
        print(f'Remove {len(files)} files with suffix {suffix} at {dir}.')


def rmfiles(files, verbose=False):
    for f in files:
        rmfile(f, verbose=verbose)


def rmfile(f, verbose=False):
    if not os.path.isfile(f):
        f'{f} is not file. Skip.'
        return
    else:
        os.remove(f)
        if verbose:
            print(f'remove {f}')


def write_append(f):
    if os.path.exists(f):
        append_write = 'a'  # append if already exists
    else:
        append_write = 'w'  # make a new file if not
    return append_write


def read_file(f):
    with open(f) as f:
        cont = f.readlines()
    return cont


def pcname():
    import socket
    return socket.gethostname()


def fashion_dir():
    name = pcname()
    if name == 'u112222':
        return '/home/chen/fromosu/tbyb/'
    elif name == 'MacBook-Pro-3.local':
        return '/Users/admin/Documents/osu/Research/tbyb/'
    elif name == '88665a240f34.ant.amazon.com':
        return '/Users/checai/Documents/fashion-space-clustering/'
    elif 'ip' in name:
        make_dir('/home/ubuntu/proj/')
        return '/home/ubuntu/proj/tbyb/raw_data/'


def microsoft_rec_dir():
    name = pcname()
    if name == 'u112222':
        return '/home/chen/fromosu/Signor/signor/ml/rec/recommenders/'
    elif 'ip' in name:
        make_dir('/home/ubuntu/proj/')
        return '/home/ubuntu/proj/recommenders/'


def cur_dir():
    return 'os.path.dirname(os.path.realpath(__file__))'


def sim_datadir():
    import socket
    assert socket.gethostname() in ['u112222', 'CSE-SCSE101549D'], 'Not on UCSD/OSU computer'
    return '/data/chen/learning_to_simulate'


def safe_remove(path):
    if path is not None:
        try:
            os.remove(path)
        except OSError:
            pass

def simplifyPath(path: str) -> str:
    stack = []

    # Split the input string on "/" as the delimiter
    # and process each portion one by one
    for portion in path.split("/"):
        # If the current component is a "..", then
        # we pop an entry from the stack if it's non-empty
        if portion == "..":
            if stack:
                stack.pop()
        elif portion == "." or not portion:
            # A no-op for a "." or an empty string
            continue
        else:
            # Finally, a legitimate directory name, so we add it
            # to our stack
            stack.append(portion)

    # Stich together all the directory names together
    final_str = "/" + "/".join(stack)
    return final_str

if __name__ == '__main__':
    import os
    dir = cur_dir()
    eval(dir)
    exit()
    f = 'a/b/c/d/e/../'
    print(simplifyPath(f))
    exit()
    print(home_dir())
    exit()
    ret = all_dirs('/home/chen/Dropbox/Wei_Data/HEA_System/4_ele_fully_random_verify')
    print(ret)
    exit()
    print(home_dir())
    exit()
    f = mktemp(tmp_dir=False)
    print(f)
    exit()
    print(mat_dir())
    exit()

    dir = f'{home()}/Documents/DeepLearning/Signor/signor/'
    import pickle

    x = 1
    with open(dir + '_test.pkl', 'wb') as f:
        pickle.dump(x, f)
    exit()
    rm_files(dir=dir, suffix='txt')
    exit()
    # print(cif_sample_dir())
    # exit()
    dir = f'{home()}/Documents/DeepLearning/material/Wei/data/TianXie/cif/elasticity.K_VRH/'
    files = find_files(dir, suffix='.cif', include_dir=True)
    print(len(files))

    exit()
    print(curdir())
    exit()

    # print(mktemp(tmp_dir=True))
    # print()
    # print(mktemp(tmp_dir=False))
    # exit()

    for _ in range(2):
        tio = torch_io(verbose=True)
        data = torch.rand((10000, 5))
        tio.dump(data, 'tmp_data')
        loaded_data = tio.load('tmp_data')
        assert (data == loaded_data).all(), 'Fail ioio unittest'
    exit()

    dir = '/Users/admin/Documents/osu/Research/Signor/data_/autoencoder'
    pprint(allf(dir, name_only=False))

    pprint(allf(dir, name_only=True))
    exit()
