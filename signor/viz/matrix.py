# from importlib import reload  # Python 3.4+ only.
import matplotlib

matplotlib.use('tkagg')
import numpy as np
import matplotlib.pyplot as plt
from sklearn import manifold
from scipy.spatial.distance import cdist
from signor.monitor.time import timefunc
import scipy.sparse.linalg as lg
from signor.monitor.probe import summary


def matshow(x, var_name='x'):
    """ improved matshow for better height/width ratio """
    assert isinstance(x, np.ndarray)
    summary(x)
    plt.imshow(x)
    if x.shape[0] > x.shape[1] * 10:
        plt.axes().set_aspect('auto')  # http://bit.ly/37V4xL0
    plt.colorbar()
    plt.title(f'{var_name} {np.shape(x)}')
    plt.show()


def viz_matrix(m, title=''):
    # https://stackoverflow.com/questions/42116671/how-to-plot-a-2d-matrix-in-python-with-colorbar-like-imagesc-in-matlab/42116772
    # matrix = np.random.random((50,50))
    print('viz matrix of size (%s %s)' % np.shape(m))
    plt.imshow(m)
    plt.colorbar()
    plt.title(title)
    plt.show()


def stat_matrix(m):
    print('mean of the matrix is %s' % np.mean(m))


def viz_eigen(m, start=0, end=1):
    # TODO: add log scale
    w, v = lg.eigs(m, k=end)
    w = list(w)
    w.sort(reverse=True)
    print(w[start:end])
    plt.plot(w[start:end])
    plt.title('Eigen Decay')
    plt.show()


def viz_eigenvector(v):
    plt.plot(v)
    plt.title('eigen vector')
    plt.show()


def test_remote_plot():
    import numpy as np
    import matplotlib.pyplot as plt

    # Create data_
    N = 500
    x = np.random.rand(N)
    y = np.random.rand(N)
    colors = (0, 0, 0)
    area = np.pi * 3

    # Plot
    plt.scatter(x, y, s=area, c=colors, alpha=0.5)
    plt.title('Scatter plot pythonspot.com')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()


@timefunc
def viz_distm(m, rs=42, mode='mds', y=None):
    """ Viz a distance matrix using MDS """
    # points = np.random.random((50, 2))
    # m = cdist(points, points, metric='euclidean')
    # TODO: add legend
    if mode == 'mds':
        mds = manifold.MDS(dissimilarity='precomputed', n_jobs=-1, random_state=rs, verbose=0)
        pos = mds.fit_transform(m)
    elif mode == 'tsne':
        tsne = manifold.TSNE(metric='precomputed', verbose=0, random_state=rs)
        pos = tsne.fit_transform(m)
    else:
        raise Exception('No such visualization mode')

    plt.scatter(pos[:, 0], pos[:, 1], c=y, s=0.1)
    plt.title('%s viz of matrix of size (%s %s)' % (mode, m.shape[0], m.shape[1]))
    plt.show()


def embedding_plot(mds_pos, tsne_pos, y_color):
    mds_pos = np.array(mds_pos)
    y_color = np.array(y_color)
    assert mds_pos.shape[1] == 2

    fig = plt.figure()
    ax = fig.add_subplot(211)
    ax.scatter(np.array(mds_pos[:, 0]), np.array(mds_pos[:, 1]), c=y_color, s=2)  # s = 2
    fig.subplots_adjust(top=0.8)
    print(('Type of pos is %s. Shape of pos is %s' % (type(mds_pos), np.shape(mds_pos))))

    ax = fig.add_subplot(212)
    ax.scatter(tsne_pos[:, 0], tsne_pos[:, 1], c=y_color, s=1.2)
    return fig


def graphsimport():
    np.set_printoptions(precision=4)
    import matplotlib
    matplotlib.use('Agg')
    # matplotlib.use('Agg')


def color_map(i):
    if i == 1:
        return 0.1
    if i == 0:
        return 'b'
    if i == 2:
        return 0.6
    if i == 3:
        return 0.9
    elif i == -1:
        return 'r'


def mds(m, y, rs=42):
    # TODO test
    """
    :param m: distance matrix (np.array of shape (n, n))
    :param y: label y = np.array([1] * 5 + [-1] * 5)
    :param rs: random seed
    :return: 
    """
    # input is the distance matrix
    # ouput: draw the mds/tsne 2D embedding

    assert m.shape == (len(y), len(y))
    mds = manifold.MDS(dissimilarity='precomputed', n_jobs=-1, random_state=rs, verbose=0)
    tsne = manifold.TSNE(metric='precomputed', verbose=0, random_state=rs)
    mds_pos = mds.fit_transform(m)
    tsne_pos = tsne.fit_transform(m)

    y_color = y
    assert len(y_color) == len(y)
    fig = embedding_plot(mds, tsne_pos, y_color)
    print(fig)


def plot_matrix_with_ticker(distm, tickers, title='Pairwise Element Distance', show=True, dir=None):
    """
    plot the distance matrix with tickers. Used in element distance
    :param distm:
    :param tickers:
    :return:
    """
    fig, ax = plt.subplots()

    im = ax.imshow(distm + distm.T)

    # We want to show all ticks...
    ax.set_xticks(np.arange(len(tickers)))
    ax.set_yticks(np.arange(len(tickers)))
    # ... and label them with the respective list entries
    ax.set_xticklabels(tickers)
    ax.set_yticklabels(tickers)

    # ax.colorbar()
    fig.colorbar(im)
    ax.set_title(title)
    if show: plt.show()
    if dir is not None:
        print(f'Save fig at {dir}')
        plt.savefig(dir, bbox_inches='tight')


def ellipsoid_plot():
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    import numpy as np

    phi = np.linspace(0, 2 * np.pi, 256).reshape(256, 1)  # the angle of the projection in the xy-plane
    theta = np.linspace(0, np.pi, 256).reshape(-1, 256)  # the angle from the polar axis, ie the polar angle
    radius = 4

    # Transformation formulae for a spherical coordinate system.
    x = radius * np.sin(theta) * np.cos(phi)
    y = radius * np.sin(theta) * np.sin(phi)
    z = radius * np.cos(theta)

    fig = plt.figure()  # Square figure
    # ax = fig.add_subplot(111, projection='3d')
    ax = fig.gca(projection='3d')
    ax.plot_surface(x, y, z, color='blue')
    plt.show()

if __name__ == '__main__':
    pass
    plt.show()
    exit()
    ellipsoid_plot()
    exit()
    points = np.random.random((50, 2))
    m = cdist(points, points, metric='euclidean')
    mds(None, None)
