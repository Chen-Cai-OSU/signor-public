""" load saved trajectory and perform dim reduction """
import os
import sys
from functools import partial
from textwrap import wrap

import matplotlib
import matplotlib.colors as clr
import matplotlib.pyplot as plt
import numpy as np
import torch
import umap
from sklearn import manifold
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from signor.format.format import coordinates
from signor.monitor.stats import stats
from signor.monitor.time import timefunc
from signor.viz.kwargs.util import viz_kwargs

random_seed = 42
np.random.seed(random_seed)
EMB_DIR = '/home/cai.507/Documents/DeepLearning/material/ode/augmented-neural-odes/emb/'
tf = partial(timefunc, threshold=5)


# matplotlib.use('tkagg')

def get_square_aspect_ratio(plt_axis):
    return np.diff(plt_axis.get_xlim())[0] / np.diff(plt_axis.get_ylim())[0]


def load_emb(file='test_traj.pt'):
    out = torch.load(file)
    return out


def test_color():
    import numpy as np
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    N = 23  # Number of labels

    # setup the plot
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    # define the data_
    x = np.random.rand(1000)
    y = np.random.rand(1000)
    tag = np.random.randint(0, N, 1000)  # Tag each point with a corresponding label

    # define the colormap
    cmap = plt.cm.jet
    # extract all colors from the .jet map
    cmaplist = [cmap(i) for i in range(cmap.N)]
    # create the new map
    cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)

    # define the bins and normalize
    bounds = np.linspace(0, N, N + 1)
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

    # make the scatter
    scat = ax.scatter(x, y, c=tag, s=np.random.randint(100, 500, N), cmap=cmap, norm=norm)
    # create the colorbar
    cb = plt.colorbar(scat, spacing='proportional', ticks=bounds)
    cb.set_label('Custom cbar')
    ax.set_title('Discrete color mappings')
    plt.show()
    exit()


class dim_reducer():
    def __init__(self, dir=EMB_DIR):
        self.dir = dir
        pass

    def load_minibatch(self, file, t=3, verbose=False):
        """ load data_ """
        data = torch.load(self.dir + file)

        # check data_ shape
        data, target = data[0], data[1]
        # print(data_.size()) # pt.Size([100, 128, 1, 28, 28])
        n_minibs = data.size()[1]
        data = data[t, :]
        data = data[:, -1]
        data = data.cpu().detach().numpy()
        data = data.reshape(n_minibs, -1)

        target = target.cpu().detach().numpy()

        if verbose:
            print(f'data_ shape is {data.shape}')
            print(f'target shape is {target.shape}')

        return data, target

    @tf
    def load_data(self, t=3, verbose=False, all=True):
        """

        :param t:
        :param verbose:
        :param all:
        :return:
        """
        files = [name for name in os.listdir(EMB_DIR) if (os.path.isfile(EMB_DIR + name) and 'mnist' in name)]
        inputs, outputs = [], []
        print('Loading data_...')
        if not all: files = files[:15]
        for file in files:
            x, target = self.load_minibatch(file, t=t, verbose=verbose)
            inputs.append(x)
            outputs.append(target)

        x = np.concatenate(inputs)
        target = np.concatenate(outputs)

        if verbose:
            print(f'data_ shape is {x.shape}')
            print(f'target shape is {target.shape}')

        return x, target

    def sampler(self, data, s=1000, verbose=False):
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

    @tf
    def acc(self, x, target, clf='lr', verbose=False):
        """
        :param x: np.array of (n, d)
        :param target: np.array of (n, )
        :return:
        """
        if verbose:
            print(f'x is of shape {x.shape}')
            print(f'target is of shape {target.shape}')
            print(f'Total number of class is {len(set(target.tolist()))}')

        x_train, x_test, y_train, y_test = train_test_split(x, target, test_size=.5, random_state=42)

        # linear clf
        print(f'Classifier: {clf}.', end='\t')
        if clf == 'lr':
            clf = LogisticRegression(random_state=0)
        elif clf == 'svm_linear':
            clf = SVC(kernel='linear')
        elif clf == 'svm_rbf':
            clf = SVC(gamma='auto', C=1000)
        else:
            raise (f'no clf {clf}')

        clf.fit(x_train, y_train)
        y_pred_test = clf.predict(x_test)
        accuracy = np.sum(y_pred_test == y_test) / y_test.shape[0]
        print(f'training score is {clf.score(x_train, y_train)}. \t test accuracy is {accuracy}')

    def assert_shape(self, x):
        if isinstance(x, torch.Tensor):
            try:
                x = x.numpy()
            except RuntimeError:  # Can't call numpy() on Variable that requires grad. Use var.detach().numpy() instead.
                x = x.detach().numpy()

        assert isinstance(x, np.ndarray)
        assert len(x.shape) == 2
        assert x.shape[1] > 2, f'data_ shape is {x.shape}'
        return x

    @tf
    def mds(self, x, kwargs=None):
        x = self.assert_shape(x)
        if kwargs is None: kwargs = {'dissimilarity': 'euclidean', 'n_jobs': -1}
        mds = manifold.MDS(**kwargs, random_state=random_seed, verbose=0)
        pos = mds.fit_transform(x)
        return pos

    @tf
    def tsnes(self, x, target):
        kwargs_list = viz_kwargs(method='tsne').load_kwargs()
        pos_list = []
        for kwargs in kwargs_list:
            pos = self.tsne(x, kwargs=kwargs)
            pos_list.append(pos)
        axarr = self.plot_images(pos_list, kwargs_list, target=target, method='tsne')
        plt.show()

    @tf
    def umaps(self, x, target):
        kwargs_list = viz_kwargs(method='umap').load_kwargs()
        pos_list = []
        for kwargs in kwargs_list:
            pos = self.umap(x, kwargs=kwargs)
            pos_list.append(pos)
        axarr = self.plot_images(pos_list, kwargs_list, target=target, method='umap')
        plt.show()

    def plot_images(self, pos_list, kwargs_list, target=None, method='tsne'):
        """ helper function for plotiing multiple tsne/umap of different hyperparameters """

        assert len(pos_list) == len(kwargs_list)
        if method == 'tsne':
            n_row, n_col = 2, 3
        elif method == 'umap':
            n_row, n_col = 5, 6
        else:
            exit(f'visualization method {method} is not supported')
        fig, axarr = plt.subplots(n_row, n_col, figsize=(n_col * 3, n_row * 3), dpi=1000)
        assert n_row * n_col >= len(pos_list)

        if target is not None:
            assert len(target) == len(pos_list[0])
            cmap, norm, bounds = self.set_color(target, discrete=True)

        for i in range(len(pos_list)):
            if i >= len(pos_list): continue
            x, y = coordinates(idx=i, n_col=n_col, n_row=n_col)
            pos = pos_list[i]

            if target is not None:
                axarr[x][y].scatter(pos[:, 0], pos[:, 1], alpha=1, s=10, linewidths=0, c=target, cmap=cmap, norm=norm)
            else:
                axarr[x][y].scatter(pos[:, 0], pos[:, 1], alpha=1, s=10, linewidths=0)

            axarr[x][y].tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False,
                                    left=False, labelleft=False)
            title = kwargs_list[i]
            axarr[x][y].set_title("\n".join(wrap(str(title))), fontsize=5)  # todo: pass axarr around
        return axarr

    @tf
    def tsne(self, x, kwargs=None):
        x = self.assert_shape(x)
        if kwargs is None: kwargs = {'metric': 'euclidean'}
        tsne = manifold.TSNE(verbose=0, random_state=random_seed, **kwargs)
        pos = tsne.fit_transform(x)
        return pos

    @tf
    def pca(self, x):
        x = self.assert_shape(x)
        pca = PCA(n_components=2)
        pos = pca.fit_transform(x)
        return pos

    @tf
    def umap(self, x, kwargs=None):
        x = self.assert_shape(x)
        if kwargs is None:
            kwargs = {'n_neighbors': 15, 'min_dist': 0.1, 'n_components': 2, 'metric': 'euclidean'}

        reducer = umap.UMAP(**kwargs, random_state=random_seed)  # todo tune these parameters
        pos = reducer.fit_transform(x)
        return pos

    def viz_x(self, x, idx=1):
        """
        :param x: np.array of shape (n, 784)
        :param idx: image id
        :return:
        """
        assert x.shape[1] == 784, f'size of x is {x.shape}, not expected'
        img = x[idx, :]
        img = img.reshape((28, 28))

        plt.imshow(img)  # todo: check rescale
        plt.show()

    def set_color(self, target, discrete=True):
        """
        :param target:
        :param discrete: if target is for classification, true. otherwise false.
        :return: cmap, norm, bounds
        """

        assert isinstance(target, np.ndarray)

        # https://stackoverflow.com/questions/12487060/matplotlib-color-according-to-class-labels
        N = len(set(target.tolist()))

        # define the colormap
        if discrete:
            cmap = plt.cm.jet
        else:
            # cmap = clr.LinearSegmentedColormap.from_list('custom blue', ['#244162', '#DCE6F1'], N=256)
            cmap = clr.LinearSegmentedColormap.from_list('custom blue', ['red', 'blue'], N=256)

        if discrete:
            # extract all colors from the .jet map
            cmaplist = [cmap(i) for i in range(cmap.N)]
            # create the new map
            cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)

            # define the bins and normalize
            bounds = np.linspace(0, N, N + 1)
            norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
            return cmap, norm, bounds
        else:
            return cmap, None, None

    def set_color_cont(self, target):
        # return
        # https://stackoverflow.com/questions/37401872/custom-continuous-color-map-in-matplotlib
        some_matrix = (np.random.rand(20, 20) - 1) * 100
        pts = np.random.rand(200, 2)
        c = (np.random.rand(200, ) - 1) * 100

        cmap = clr.LinearSegmentedColormap.from_list('custom blue', ['#244162', '#DCE6F1'], N=256)

        # plt.matshow(some_matrix, cmap=cmap)
        plt.scatter(pts[:, 0], pts[:, 1], c=c, cmap=cmap)
        plt.colorbar()
        plt.show()
        pass

    def show_emb(self, emb, title=None, axarr=None, show=True, coor=(1, 1)):
        """
        :param emb: np.array of shape (n, 2)
        :param title:
        :param axarr: an array of <matplotlib.axes._subplots.AxesSubplot>
        :param show: if True, plt.show()
        :param coor: subplot coordinates
        :return: No return
        """
        assert isinstance(emb, np.ndarray)
        assert emb.shape[1] == 2

        if axarr is None:
            plt.scatter(emb[:, 0], emb[:, 1])
            plt.title(title)
        else:
            x, y = coor
            assert isinstance(axarr, np.ndarray)  # an array of <matplotlib.axes._subplots.AxesSubplot>
            axarr[x][y].scatter(emb[:, 0], emb[:, 1])
            axarr[x][y].set_title(title)

        if show:
            plt.show()


import argparse

parser = argparse.ArgumentParser(description='Dim Reduction')
parser.add_argument('--t', default=3, type=int, help='morph from 0 to 1. t/100.')

if __name__ == '__main__':
    from signor.datasets.image_classification.cifar import load_cifar

    args = parser.parse_args()

    dr = dim_reducer()
    dr.set_color_cont(None)
    exit()
    train, _, = load_cifar(data_dir='./', nr_classes=10)
    # train, _, _ = load_mnist(data_dir='./')
    x, target = train
    x = x.reshape(50000, -1)
    x, target = dr.sampler((x, target), s=1000)

    dr.tsnes(x, target)
    dr.umaps(x, target)
    sys.exit()

    # plot
    n_plots, n_row, n_col = 4, 2, 2
    fig, axarr = plt.subplots(n_row, n_col, figsize=(20, 10))
    print(axarr)

    alpha = 1
    cmap, norm, bounds = dr.set_color(target, discrete=True)
    viz_mtds = ((0, 'pca', dr.pca), (1, 'mds', dr.mds), (2, 'tsne', dr.tsne), (3, 'pca', dr.pca))

    for i in range(n_plots):
        row, col = i // n_row, i % n_row

        emb_2d = viz_mtds[i][2](x)  # dr.pca(x)
        axarr[row][col].scatter(emb_2d[:, 0], emb_2d[:, 1], alpha=alpha, linewidths=0, c=target, cmap=cmap, norm=norm)

        axarr[row][col].tick_params(axis='both', which='both', bottom=False,
                                    top=False, labelbottom=False, right=False,
                                    left=False, labelleft=False)
        # axarr[row][col].set_aspect(get_square_aspect_ratio(axarr[row][col]))
        axarr[row][col].set_title(f'methods: {viz_mtds[i][1]}')

    plt.show()
