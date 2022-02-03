# Created at 2021-04-08
# Summary:
import random

import matplotlib.pyplot as plt
import numpy as np

random.seed(42)


def space_tmp_svd(data, dim, n):
    assert dim * n == data.shape[0]
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(6, 5))

    axes[0].set_yscale('log')
    for interval in [1, 2, 4, 6, 8, 10, 14]:
        atom_indieces = random.sample(range(n), n // interval)
        indices = []
        for idx in atom_indieces:
            indices += list(range(idx * dim, (idx + 1) * dim))

        s_interval = np.linalg.svd(data[indices, :], full_matrices=True, compute_uv=False)
        axes[0].plot(s_interval, label=interval)
    axes[0].legend()
    axes[0].set_title('sample space')

    axes[1].set_yscale('log')
    for interval in [1, 2, 4, 6, 8, 10, 14]:
        s_interval = np.linalg.svd(data[:, ::interval], full_matrices=True, compute_uv=False)
        axes[1].plot(s_interval, label=interval)
    axes[1].legend()
    axes[1].set_title('sample time')

    fig.tight_layout()
    fig.suptitle(f'data: {data.shape}')
    plt.show()
