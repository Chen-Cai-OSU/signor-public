import numpy as np
from signor.monitor.stats import stats

random_seed = 42

# copyed from dim-red.py
def sampler(data, s=1000, verbose=False):
    """
    :param data: (np.array of shape (n, d), np.array of shape (n, )) or just np.array of shape (n, d)
    :param s:
    :return:
    """
    if isinstance(data, list) or isinstance(data, tuple):
        x, target = data
    else:
        assert isinstance(data, np.ndarray)
        x = data
        target = np.zeros((x.shape[0],))

    n = x.shape[0]
    assert target.shape == (n,)

    if s > n:
        print(f'sample size {s} larger than data_ size {n}. Round s to {n}')
        s = n

    np.random.seed(random_seed)
    indices = np.random.choice(n, size=s, replace=False).tolist()

    x_, target_ = x[indices, :], target[indices]

    if verbose:
        print('sample indices stats:')
        stats(indices, precision=2, verbose=True)
        print(f'sample data_ shape is {x_.shape}')
        print(f'sample target shape is {target_.shape}')

    return x_, target_

if __name__ == '__main__':
    x, y = np.random.random((1000, 10)), np.random.random((1000,))
    data = (x, y)
    sampler(data, s=20, verbose=True)