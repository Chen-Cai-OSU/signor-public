import numpy as np
from tqdm import tqdm

from signor.datasets.o3 import rot
import os.path as osp

from signor.ioio.dir import sig_dir


def plot3d():
    import matplotlib.pyplot as plt
    import numpy as np

    def randrange(n, vmin, vmax):
        '''
        Helper function to make an array of random numbers having shape (n, )
        with each number distributed Uniform(vmin, vmax).
        '''
        return (vmax - vmin) * np.random.rand(n) + vmin

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    n = 100

    # For each set of style and range settings, plot n random points in the box
    # defined by x in [23, 32], y in [0, 100], z in [zlow, zhigh].
    for c, m, zlow, zhigh in [('r', 'o', -50, -25), ('b', '^', -30, -5)]:
        xs = randrange(n, 23, 32)
        ys = randrange(n, 0, 100)
        zs = randrange(n, zlow, zhigh)
        ax.scatter(xs, ys, zs, c=c, marker=m)

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.show()

import matplotlib.pyplot as plt
import numpy as np

def plot_ellipsoid(rot, n=100,  title=None):
    """
    plot psd matrix as an ellipsoid;
    used for permeability paper; the other function plotting morse graphs
     is at graph/permeability/util/cmp.py
     """

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection='3d')

    coefs = (0.30, 0.59, 0.74) # eigenvalues
    rx, ry, rz = coefs

    # Set of all spherical angles:
    u = np.linspace(0, 2 * np.pi, n)
    v = np.linspace(0, np.pi, n)

    # Cartesian coordinates that correspond to the spherical angles:
    x = rx * np.outer(np.cos(u), np.sin(v))
    y = ry * np.outer(np.sin(u), np.sin(v))
    z = rz * np.outer(np.ones_like(u), np.cos(v))

    xyz = np.concatenate([x.reshape(-1, 1), y.reshape(-1, 1), z.reshape(-1, 1)], axis=1)
    xyz = xyz @ rot.T
    xrot, yrot, zrot = xyz[:, 0].reshape(n, n), xyz[:, 1].reshape(n, n), xyz[:, 2].reshape(n, n)

    # Plot:
    params = {}  # {rstride:4, cstride:4, }
    ax.plot_surface(xrot, yrot, zrot, cmap=plt.cm.Spectral, **params)

    # Adjustment of the axes, so that they all have the same span:
    max_radius = max(rx, ry, rz)
    for axis in 'xyz':
        getattr(ax, 'set_{}lim'.format(axis))((-max_radius, max_radius))

    params={'fontsize': 30}
    ax.set_xlabel('$X$', **params)
    ax.set_ylabel('$Y$', **params)
    ax.set_zlabel('$Z$', **params)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    # plt.xticks([], [])
    # plt.yticks([], [])
    # plt.zticks([], [])

    f = osp.join(sig_dir(), 'graph', 'permeability', 'paper', 'viz', str(title)+'.pdf')
    plt.savefig(f, bbox_inches='tight')


if __name__ == '__main__':
    n_img = 5
    angle = np.pi/2
    for i in tqdm(range(n_img )):
        rotation = rot(0, angle/ n_img * i, 0).numpy()
        plot_ellipsoid(rotation, title=f'ellipsoid-y-{i}')

    for i in tqdm(range(n_img )):
        rotation = rot(angle/ n_img * i, 0, 0).numpy()
        plot_ellipsoid(rotation, title=f'ellipsoid-z-{i}')
