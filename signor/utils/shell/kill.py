# used for killing certain process
# for pid in $(ps -ef | grep "fcc" | awk '{print $2}'); do kill -9 $pid;

import argparse
# for pid in $(ps -ef | grep "fcc" | awk '{print $2}'); do echo $pid;
import os

from signor.format.format import banner
from signor.ioio.keyboard import yes_or_no


def kill(search='fcc'):
    banner(f'Kill {search} related cmd')
    echo_cmd = f'for pid in $(ps -ef | grep "{search}" | awk \'{{print $2}}\'); do echo $pid; done'
    kill_cmd = f'for pid in $(ps -ef | grep "{search}" | awk \'{{print $2}}\'); do kill -9 $pid; done'
    print(echo_cmd)
    print(kill_cmd)

    verbose_cmd = f'ps -ef | grep "{search}"'
    os.system(verbose_cmd)
    if yes_or_no(f'kill all processes for {search}?'):
        os.system(kill_cmd)


parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, default='subgraph', help='file name to kill')
if __name__ == '__main__':
    args = parser.parse_args()
    kill(search=args.name)
