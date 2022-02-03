# Created at 2020-04-22
# Summary: system related util


import os
n=10
os.environ['MKL_NUM_THREADS'] = str(n)
os.environ['OMP_NUM_THREADS'] = str(n)
os.environ['OPENBLAS_NUM_THREADS'] = str(n)
os.environ['MKL_NUM_THREADS'] = str(n)
os.environ['VECLIB_MAXIMUM_THREADS'] = str(n)
os.environ['NUMEXPR_NUM_THREADS'] = str(n)
import torch
torch.set_num_threads(n) # always import this first
status = f'{n}'


import platform

def detect_sys():
    sys = platform.system()

    if sys == 'Linux':
        return sys
    elif sys == 'Darwin':
        return sys
    else:
        NotImplementedError

if __name__ == '__main__':
    sys = detect_sys()
    import numpy as np
    from time import sleep
    n = 10000
    x, y = np.random.random((n, n)), np.random.random((n, n))
    z = x@y
    print(sys)
