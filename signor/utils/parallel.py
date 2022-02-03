from multiprocessing import Process
import sys
import os
import multiprocessing
from signor.monitor.time import timefunc
from time import sleep
import numpy as np
from joblib import delayed, Parallel

rocket = 0
max_n = 10


@timefunc
def func1():
    global rocket
    print('start func1')
    while rocket < max_n:
        rocket += 1
    sleep(10)
    print('end func1')


@timefunc
def func2():
    global rocket
    print('start func2')
    while rocket < max_n:
        rocket += 1
    sleeptime = np.random.random_integers(0, 10)
    sleep(sleeptime)
    print(f'end func2 where sleep time is {sleeptime}')


from joblib import Parallel, delayed


def parallel_cmds():
    cmds = []
    for i in range(10):
        cmd = f'sleep 2 || echo "finished {i}" & '
        cmds.append(cmd)

    for cmd in cmds:
        # print(cmd)
        os.system(cmd)


class test_parallel:
    """
    http://qingkaikong.blogspot.com/2016/12/python-parallel-method-in-class.html
    """

    def __init__(self):
        pass

    def run(self, cmd):
        assert isinstance(cmd, str)
        os.system(cmd)

    def run_parallel(self, cmds, n_jobs=1):
        Parallel(n_jobs=n_jobs)(delayed(unwrap_self)(cmd) for cmd in cmds)

    def fancy_run_parallel(self):
        cmd = 'sleep 1'
        cmds = [cmd] * 40
        self.run_parallel(cmds)


def unwrap_self(cmd):
    return test_parallel().run(cmd)

import torch

def torch_parallel(model, parallel=False, devices_id=[0,1]):
    if parallel and torch.cuda.device_count() > 1:
        # https://pytorch.org/tutorials/beginner/blitz/data_parallel_tutorial.html
        print("There are", torch.cuda.device_count(), "GPUs available")
        print(f"Use devices: {devices_id}")
        model = torch.nn.DataParallel(model, device_ids=devices_id)
        return model
    else:
        # todo: add GPU id
        # https://discuss.pytorch.org/t/how-to-check-if-model-is-on-cuda/180
        machine = 'GPU' if next(model.parameters()).is_cuda else 'CPU'
        print(f'Use (single) {machine}')
        return model




if __name__ == '__main__':
    # test_parallel().fancy_run_parallel()

    # exit()
    cmd = 'sleep 3'
    cmds = [cmd] * 40
    for _ in range(3):
        test_parallel().run_parallel(cmds, n_jobs=20)

    exit()
    from multiprocessing import Pool
    import subprocess


    def run_command(t):
        command = f"bash sleep {t}. "
        subprocess.Popen(command, shell=True)


    pool = Pool()
    times = [3] * 10
    pool.map(run_command, times)
    exit()

    parallel_cmds()
    exit()
    try:
        Parallel(n_jobs=-1, timeout=20)(delayed(func2)() for _ in range(5))
    except multiprocessing.context.TimeoutError:
        print('Time out Error')
    exit()

    p1 = Process(target=func1)
    p1.start()
    p2 = Process(target=func2)
    p2.start()
