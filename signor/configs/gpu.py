import os
import random
import subprocess
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from pynvml import nvmlDeviceGetHandleByIndex, nvmlInit, nvmlDeviceGetMemoryInfo

from signor.format.format import red
from signor.monitor.time import timefunc, curtime


class gpu_monitor():
    def __init__(self):
        assert torch.cuda.is_available(), 'no gpu found'

    @staticmethod
    def query_mem(self, device=0, verbose=0):
        nvmlInit()
        handle = nvmlDeviceGetHandleByIndex(device)
        info = nvmlDeviceGetMemoryInfo(handle)
        mem_const = 1024 ** 2.0

        if verbose == 1:
            print(f"Total memory MB: {info.total / (mem_const)}")
            print(f"Free memory MB: {info.free / mem_const}")
            print(f"Used memory MB: {info.used / mem_const} \n")

        return info.free / mem_const

    @timefunc
    def query_ava(self, device=0, t=5, mem_lim=1000):
        """
        query device if has more than mem_limit memory during t seconds
        :param device:
        :param t:
        :param mem_lim:
        :return:
        """
        for i in range(int(t)):
            free_mem = self.query_mem(device=device)
            time.sleep(1)
            if free_mem < mem_lim:
                print(f'device {device} has less than {mem_lim} MB memory available')
                return False

        print(f'device {device} has more than {mem_lim} MB memory available for {t} seconds')
        return True

    def scheduler(self, cmds=[], t=5, mem_limit=1000, n_process=10):
        """
        :param cmds:
        :param t:
        :param mem_limit:
        :param n_process:
        :return:
        """
        # todo: need to refactor
        print(f'Start to schedule {len(cmds)} cmds where \
                buffer time is {t} mem_limit is {mem_limit} MBs \
                and n_process limit is {n_process}')

        buffer_num = 0
        while len(cmds) > 0:
            buffer_time = 1 if buffer_num < 4 else t
            query_kwargs = {'mem_limit': mem_limit, 't': buffer_time}
            if get_process_num() < n_process and self.query_ava(device=0, **query_kwargs) and self.query_ava(device=1,
                                                                                                             **query_kwargs):
                print(f'assgin a new taks. in total {get_process_num()}')
                cmd = cmds.pop()  # ' nohup ~/anaconda3/bin/python Wei/baseline/GCN.py --scheduler --n_epoch 500 --bs 8 --n_data 200 --n_conv 2 --data_ mp &'
                print(cmd)
                print('-' * 50)
                print()
                os.system(cmd)
                buffer_num += 1
            else:
                time.sleep(5)
                print(f'exit mem limit or n_process limit. wait for 5 seconds')


def random_device():
    if random.random() < 0.5:
        return '0'
    else:
        return '1'


def get_process_num(cgcnn=False):
    """ count the totoal num of GCN.py process """
    # todo: knew the device identity of each process

    cmd = ' ps -ef | grep \'GCN.py\''
    output = subprocess.check_output(cmd, shell=True)  # # http://bit.ly/36meoJ0
    output = str(output)
    cnt1 = output.count('Wei/baseline/GCN.py')

    cmd = ' ps -ef | grep \'code/main.py\''
    output = subprocess.check_output(cmd, shell=True)  # # http://bit.ly/36meoJ0
    output = str(output)
    cnt2 = output.count('code/main.py')

    return cnt1 + cnt2


def query_mem(device=0, verbose=0):
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(device)
    info = nvmlDeviceGetMemoryInfo(handle)
    mem_const = 1024 ** 2.0

    if verbose == 1:
        print(f"Total memory MB: {info.total / (mem_const)}")
        print(f"Free memory MB: {info.free / mem_const}")
        print(f"Used memory MB: {info.used / mem_const} \n")

    return info.free / mem_const, info.used / mem_const


def mem_ts(device=0, times=5, plot=True, step=1):
    """
    check the gpu memory over time
    :param device: gpu device
    :param times: num of query times
    :param plot: plot or not
    :param step: sleep during each query time
    :return:
    """

    frees, useds, ts = [], [], []
    for _ in range(times):
        free, used = query_mem(device=device, verbose=0)
        frees.append(free)
        useds.append(used)
        t = curtime().split(' ')[-1]
        ts.append(t)
        time.sleep(step)

    if plot:
        plt.xticks(rotation=70)
        plt.plot(ts, frees, label='free')
        plt.plot(ts, useds, label='used')
        plt.legend()
        plt.show()

    return frees, useds, ts


def free_gpu(n_machine=1):
    # https://bit.ly/3dBmapU
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
    memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
    print(memory_available)
    ret = np.array(memory_available).argsort()[-n_machine:][::-1]  # https://bit.ly/388di7I
    prev_ret = np.argmax(memory_available)
    print(ret, prev_ret)

    if n_machine > 1:
        assert len(ret) > 1
        return ret
    else:
        assert len(ret) == n_machine == 1
        return ret[0]


def random_wait(n=30):
    """ randomly wait for n seconds """
    import numpy
    if numpy.random.random() > 0.5:
        print(red(f'GPU: Wait for {n} seconds...'))
        time.sleep(n)
    return


def bestdev(dev):
    if dev == 'cpu':
        return dev
    elif dev.startswith('cuda'):
        random_wait()
        return least_load_gpu2()
    else:
        raise NotImplementedError


def most_frequent(List):
    from collections import Counter
    occurence_count = Counter(List)
    return occurence_count.most_common(1)[0][0]


def least_load_gpu():
    import GPUtil
    candidate_gpus = []
    for _ in range(10):
        deviceIDs = GPUtil.getAvailable(order='memory', limit=1, maxLoad=0.8, maxMemory=0.8, includeNan=False,
                                        excludeID=[], excludeUUID=[])
        time.sleep(0.5)
        candidate_gpus += deviceIDs
        print(f'{candidate_gpus}')
    return most_frequent(candidate_gpus)


def least_load_gpu2():
    t0 = time.time()
    import GPUtil
    try:
        deviceID = GPUtil.getFirstAvailable(order='load', maxLoad=0.8, maxMemory=0.8, attempts=30, interval=1,
                                            verbose=False)
        print(f'Took {int(time.time() - t0)}s to find cuda:{deviceID}')
        return f'cuda:{deviceID[0]}'
    except RuntimeError:
        print('Unable to find good cuda device. Switch to cpu instead.')
        return 'cpu'


def n_gpu():
    return torch.cuda.device_count()


def random_cuda_dev_prefix():
    if not torch.cuda.is_available():
        print(red('no gpus availble'))
        return ''
    # todo: only assign idle cuda devices
    idx = np.random.randint(0, high=n_gpu())
    return f'CUDA_VISIBLE_DEVICES={idx} '


def dummy_workload(n=1000, iter=100, dev='cuda:1'):
    for _ in range(iter):
        x = torch.rand((n, n), device=dev)
        x @ x
        time.sleep(1)
    return


if __name__ == '__main__':
    print(query_mem(verbose=1))
    exit()
    print(bestdev('cuda:-1'))
    print(free_gpu(n_machine=1))
    exit()
    # pprint(mem_ts(plot=True, times=3, step=1))
    # exit()


    out = get_process_num()
    print(out)
    exit()

    GM = gpu_monitor()
    GM.query_ava(device=0)

    GM.query_ava(device=0, mem_limit=10000)

    GM.query_ava(device=0, mem_limit=100000)

    unused = query_mem(device=0)
    print(f'unused is {unused} MB')

    unused = query_mem(device=1)
    print(f'unused is {unused} MB')
