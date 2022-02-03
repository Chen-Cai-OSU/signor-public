""" clear large data_ files in signor. leave only python files """

import os
import os.path as osp
import subprocess
from pprint import pprint
import sys
import traceback

from signor.format.format import banner
from signor.ioio.dir import all_dirs, assert_dir_exist


def rmLargeFiles(print_only=True, thres=10):
    model_dir = osp.join(osp.dirname(osp.abspath(__file__)), '..', '..', '')
    cmd = f'find {model_dir} -type f -size +{thres}M '
    print(cmd)
    largefiles = subprocess.check_output(cmd, shell=True)
    largefiles = str(largefiles).split('\\n')[:-1]

    for file in largefiles:
        if '.git' in file: continue
        cmd = f'rm {file}'
        print(cmd)

        if not print_only:
            os.system(cmd)
            print('\n')

def rm_dir(dir):
    cmd = f'rm -rf {dir}'
    banner(f'rm {dir}')
    os.system(cmd)

def rm_files():
    """ remove hea files """
    dir = '/home/cai.507/Dropbox/Wei_Data/HEA_System/4_ele_fully_random_result/'
    for mat in all_dirs(dir):

        assert_dir_exist(os.path.join(dir, mat, 'bcc'), warn=False)
        rm_dir(os.path.join(dir, mat, 'bcc'))
        assert_dir_exist(os.path.join(dir, mat, 'fcc'), warn=False)
        rm_dir(os.path.join(dir, mat, 'fcc'))

from signor.utils.exception import error_msg
if __name__ == '__main__':
    rm_files()
    exit()
    # print('abc')
    # print(5)
    # try:
    #     print(1/0)
    # except:
    #     pprint(stack_trace)
        # error_msg()

    rmLargeFiles(print_only=True, thres=10)
