# 07/14/2021
# memory related utils
import numpy as np

from signor.format.format import pf


def size2gb(*args):
    gb = np.prod(np.array(args)) * 8 / (1024**3)
    print(f'array of size {args}: {pf(gb, 2)}GB (float64)')
    return gb

if __name__ == '__main__':
    # size = (422000, 422000)
    # size = (49600, 847470)
    # size = (804641620, 1)
    # size = (1e6, 1e6)
    # size = (843000, 33000)
    # size = (1.1e6, 65e3)
    # size = (147179, 1427370)
    # size = (33280, 255770)
    # size = (147179, 1427370)
    size = (1054394, 128)
    size = (566302, 65768)
    size = (4000, 4000, 8, 15)
    # size = (480189, 17770)
    # size = (1699248, 422652)
    print(size2gb(*size))