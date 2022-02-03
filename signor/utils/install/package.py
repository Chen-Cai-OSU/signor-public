# Created at 2020-05-22
# Summary: utils for manage packages
import os

from signor.utils.cli import runcmd


def find_loc(package):
    cmd = f"python -c 'import {package}; print({package}.__version__)' " # f'python | import {package}'
    runcmd(cmd)

def find_version(package):
    # https://bit.ly/2XsT9TI
    cmd = f'pip freeze | grep {package}'
    runcmd(cmd)

import argparse
parser = argparse.ArgumentParser(description='sanity check')
parser.add_argument('--pkg', type=str, default='numpy', help='package name')

if __name__ == '__main__':
    args = parser.parse_args()
    pkg = args.pkg

    find_loc(pkg)
    find_version(pkg)

