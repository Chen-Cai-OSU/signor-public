import os
import subprocess
import sys
from pprint import pprint
from signor.format.format import args_print
from signor.format.format import banner

def cap_out(cmd):
    out = subprocess.check_output(cmd, shell=True)
    pprint(out)
    return out



def capture(cmd):
    proc = subprocess.Popen(cmd,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            )
    stdout, stderr = proc.communicate()
    return proc.returncode, stdout, stderr


def test(cmd):
    print(os.system(cmd))

from signor.format.format import timestamp
def runcmd(cmd, print_only=False, background = False, nohup=False, thres=50):

    if background: cmd += ' &'
    if nohup:
        assert 'nohup' not in cmd
        cmd = 'nohup ' + cmd

    banner(f'{timestamp()} Execute cmds:', compact=True)
    if len(cmd)>thres and '--' in cmd:
        args_print(cmd)
    else:
        print(cmd)

    if not print_only:
        os.system(cmd)


if __name__ == '__main__':
    code, out, err = capture([sys.executable, '/home/cai.507/Documents/DeepLearning/Signor/signor/graph/paper/PairNorm/pairnorm.py'])

    print("out: '{}'".format(out))
    print("err: '{}'".format(err))
    print("exit: {}".format(code))

    exit()
    # cmd = 'ls' # "uname -a | awk '{print $9}'"
    cmd = 'ls' #'python graph/cgcnn/code/paper/parse_results.py --neph 0 --geph 0 --xt_id 3000 --xt_idA 0'
    out = cap_out(cmd)
    print(cmd)


    # test(cmd)
    # cap_out('pwd')
