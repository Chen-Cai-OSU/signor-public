# Created at 2020-04-14
# Summary: generate script for crontab
import os

from signor.format.format import banner, timestamp
from signor.ioio.dir import log_dir, sig_dir


def gen_cmd():
    dir = os.path.join(log_dir(), 'hea', '')
    cmd = f" grep -ni 'finish' --include '*.log' {dir}*.log | wc "
    banner(f'{timestamp()}: {cmd}')
    os.system(cmd)

def exe_parallel_args():
    f = os.path.join(sig_dir(), 'utils', 'shell', 'parallel_args.py')
    cmd = f'python {f}'
    banner(f'{timestamp()}: {cmd}')
    os.system(cmd)


if __name__ == '__main__':
    gen_cmd()

    exe_parallel_args()
