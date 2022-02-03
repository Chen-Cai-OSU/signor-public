""" utils for sparse np.array """
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.sparse import coo_matrix

from signor.utils.np import np2set, tonp


def random_zero_one_array(size=(10, 10)):
    x = np.random.random(size)
    return (np.sign(x - 0.9) + 1)/2

def sparsity_plot(x):
    """ look through cols and check the sparsity of each column """
    assert isinstance(x, (np.ndarray, torch.Tensor))
    x = tonp(x)
    # x = x.astype(int)

    assert np2set(x) <= set([0, 1]), f'np2set(x) is {np2set(x)}'

    col_sparsity = np.sum(x == 0, axis=0)
    col_nonsparsity = np.sum(x != 0, axis=0)
    plt.plot(col_sparsity, label='sparsity')
    plt.plot(col_nonsparsity, label='non-sparsity')
    plt.title('Sparsity level')
    plt.legend()
    plt.show()

    _ = plt.hist(col_sparsity, bins='auto')  # arguments are passed to np.histogram
    plt.title('Distribution of sparsity across columns')
    plt.show()

def sparse_tensor2_sparse_numpyarray(sparse_tensor):
    """
    :param sparse_tensor: a COO torch.sparse.FloatTensor
    :return: a scipy.sparse.coo_matrix
    """
    if sparse_tensor.device.type == 'cuda':
        sparse_tensor = sparse_tensor.to('cpu')

    values = sparse_tensor._values().numpy()
    indices = sparse_tensor._indices()
    rows, cols = indices[0,:].numpy(), indices[1,:].numpy()
    size = sparse_tensor.size()
    scipy_sparse_mat = coo_matrix((values, (rows, cols)), shape=size, dtype=np.float)
    return scipy_sparse_mat

if __name__ == '__main__':
    x = random_zero_one_array(size=(1500, 1000))
    sparsity_plot(x)

