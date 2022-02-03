from pprint import pprint
import numpy as np
import scipy
import torch
from collections import defaultdict

np.random.seed(42)


def isndarray(arr):
    return isinstance(arr, np.ndarray)


def most_freq(x):
    isinstance(x, np.ndarray)
    (values, counts) = np.unique(x, return_counts=True)
    ind = np.argmax(counts)
    print(f'most freq value is {values[ind]} ({np.max(counts)})')


def most_freqk(x, k=1, precision=3):
    from signor.utils.list import most_freqk

    x = x.reshape((x.size,))
    res = most_freqk(x.tolist(), k=k, precision=precision)
    print(f'Most frequent {k} numbers:')
    pprint(res)


def one_hot(label, nr_classes, dtype='float32'):
    if isinstance(label, int) or (isndarray(label) and len(label.shape) == 0):
        out = np.zeros(nr_classes, dtype=dtype)
        out[int(label)] = 1
        return out

    assert len(label.shape) == 1
    nr_labels = label.shape[0]
    out = np.zeros((nr_labels, nr_classes), dtype=dtype)
    out[np.arange(nr_labels), label] = 1
    return out


def tonp(tsr):
    if isinstance(tsr, np.ndarray):
        return tsr

    assert isinstance(tsr, torch.Tensor), f'{type(tsr)}'
    tsr = tsr.cpu()
    assert isinstance(tsr, torch.Tensor)
    try:
        arr = tsr.numpy()
    except:
        arr = tsr.detach().numpy()
    assert isinstance(arr, np.ndarray)
    return arr


def totsr(arr):
    if isinstance(arr, torch.Tensor):
        return arr
    else:
        assert isinstance(arr, np.ndarray)
        return torch.Tensor(arr)


def issquare(m):
    assert isinstance(m, (np.ndarray, torch.Tensor))
    m = tonp(m)
    assert np.ndim(m) == 2, f'Shape of m is {np.shape(m)}'
    return m.shape[0] == m.shape[1]


def eig(m, **kwargs):
    """ compute eigenvalues of a np.array/tensor """
    import matplotlib.pyplot as plt
    from scipy.linalg import eigvals

    assert isinstance(m, (np.ndarray, torch.Tensor))
    m = tonp(m)

    if issquare(m):
        eigv = eigvals(m)
        _title = 'square'
    else:
        U, s, v = scipy.linalg.svd(m)
        eigv = s
        _title = 'rectangle'
    if kwargs.get('layer', None) is not None:
        _title = kwargs['layer']

    abs_eigv = np.abs(eigv)
    abs_eigv.sort()
    abs_eigv = abs_eigv[::-1]
    plt.plot(abs_eigv, label='abs')
    plt.plot(eigv, label='non-abs')
    plt.legend()
    plt.title(f'eigen decay of {_title}')
    plt.yscale('log')
    plt.show()
    return eigv


def norm(x, ord=None):
    """ compute the norm a np.array/tensor. by default 2-norm. """
    assert isinstance(x, (np.ndarray, torch.Tensor))
    x = tonp(x)
    return scipy.linalg.norm(x, ord=ord)


def add_noise(x, ndim=10):
    assert isinstance(x, np.ndarray)
    assert np.ndim(x) == 2
    n, d = x.shape
    noise = np.random.random((n, ndim))
    return np.concatenate((x, noise), axis=1)


def unique_row(data):
    assert isinstance(data, np.ndarray)
    return np.unique(data, axis=0)


def assert_rows_unique(data):
    rows = unique_row(data)
    assert rows.shape[0] == data.shape[0], f'unique rows {rows.shape}. data {data.shape}.'


def np2set(x):
    assert isinstance(x, np.ndarray)
    return set(np.unique(x))


def rm_zerocol(data, print_flag=False):
    """ remove zero columns """
    x = np.delete(data, np.where(~data.any(axis=0))[0], axis=1)

    if print_flag:
        print(f'the shape before/after removing zero columns is {np.shape(data)}/{np.shape(x)}')

    return x


def index_of_firstx(arr, x=1, not_found=-1):
    """
        http://bit.ly/2TvAGoy
        For array of shape (n, d), find the indices of first x in each row.
        return array of shape (n,). If there is not x in certain row, return -1
    """
    mask = arr == x
    ans = np.where(mask.any(1), mask.argmax(1), not_found)
    ans = ans.reshape((len(ans), 1))
    return ans


def sampler(data, s=1000, verbose=False, rs=42):
    """
    :param data: (np.array of shape (n, d), np.array of shape (n, )) or just np.array of shape (n, d)
    :param s: sample size
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

    np.random.seed(rs)
    indices = np.random.choice(n, size=s, replace=False).tolist()

    x_, target_ = x[indices, :], target[indices]

    if verbose:
        print('sample indices stats:')
        stats(indices, precision=2, verbose=True)
        print(f'sample data_ shape is {x_.shape}')
        print(f'sample target shape is {target_.shape}')

    return x_, target_


def non_zero(x):
    """ get the non-zero values of an array """
    assert isinstance(x, np.ndarray)
    return x.ravel()[np.flatnonzero(x)]


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::
    https://bit.ly/2YHzUYK

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def heatmap(x):
    import matplotlib.pyplot as plt

    if isinstance(x, torch.Tensor):
        x = x.numpy()

    if x.ndim == 1:
        x = np.expand_dims(x, axis=1)

    assert isinstance(x, np.ndarray)
    assert x.ndim == 2
    n, d = x.shape
    ratio = max(n / d, d / n)
    aspect = 'auto' if ratio > 10 else 'equal'
    plt.imshow(x, cmap='hot', interpolation='nearest', aspect=aspect)
    plt.colorbar()
    plt.show()


def non_zero_indices(x):
    # given a np.array, return a dict of indices tuples whose corresponding values is non-zero
    # works for both np.ndarray and csr_matrix
    indices = np.transpose(np.nonzero(x))
    d = defaultdict(set)
    for x, y in indices:
        d[x].add(y)
    return d


def zero_indices(x):
    # given a np.array, return a dict of indices tuples whose corresponding values is 0

    # only flip the zero entries in the csr.matrix, not all elements in the whole matrix
    if isinstance(x, scipy.sparse.csr.csr_matrix):
        tmp = scipy.sparse.csr_matrix.sign(x)
        tmp.data = 1 - tmp.data
        return non_zero_indices(tmp)
    else:
        return non_zero_indices(1 - np.sign(x))


if __name__ == '__main__':
    x = np.array([[3, 0, 0], [0, 4, 0], [5, 6, 0]])
    print(zero_indices(x))
    exit()
    x = np.random.random((1000, 3))
    heatmap(x)
    exit()
    label = np.array([1, 2, 3])
    print(one_hot(label, 4))
    exit()
    arr = np.array([[0, 1, 0, 2, 2, 1, 2, 2],
                    [1, 1, 2, 2, 2, 1, 0, 1],
                    [2, 1, 0, 1, 0, 0, 2, 0],
                    [2, 2, 1, -1, 1, 2, 1, -1]])
    print(non_zero(arr))
    exit()
    most_freq(arr)
    exit()

    assert_rows_unique(arr)
    exit()

    print(index_of_firstx(arr, x=0, not_found=-10))

    exit()
    data = np.array([[1, 2], [3, 4], [1, np.nan]])  # np.array([[1,2], [3,4], [1,2]])
    print(np2set(data))
    exit()

    x = np.random.random((100, 10))
    print(np.mean(x))
    x_noise = add_noise(x, ndim=10)
    print(x_noise.shape)
    print(np.mean(x_noise))

    exit()
    x = np.random.random((1000, 100))
    x = x.T
    from signor.monitor.stats import stats

    stats(x)
    eig(x)
    exit()

    x = np.array([[1, 2], [3, 4]])
    print(norm(x) ** 2)
    exit()
    m = np.dot(x, x.T)
    eig(m)
    exit()
