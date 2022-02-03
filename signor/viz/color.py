# https://stackoverflow.com/questions/12487060/matplotlib-color-according-to-class-labels

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from signor.monitor.probe import summary


def set_color(tag):
    # tag = np.random.randint(0, N, 1000)  # Tag each point with a corresponding label
    assert isinstance(tag, np.ndarray)
    assert tag.shape == (len(tag), )

    N = len(set(tag.tolist()))
    cmap = plt.cm.jet  # define the colormap
    cmaplist = [cmap(i) for i in range(cmap.N)]  # extract all colors from the .jet map
    cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)  # # create the new map

    # define the bins and normalize
    bounds = np.linspace(0, N, N + 1)
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    return tag, cmap, norm, bounds


if __name__ == '__main__':
    N = 23  # Number of labels

    # setup the plot
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    data = np.random.random((1000, 2))


    tag = np.random.randint(0, N, 1000)  # Tag each point with a corresponding label
    summary(tag, 'tag')
    tag, cmap, norm, bounds = set_color(tag)

    # make the scatter
    scat = ax.scatter(data[:, 0], data[:, 1], c=tag, s=np.random.randint(100, 500, N), cmap=cmap, norm=norm)
    # create the colorbar
    cb = plt.colorbar(scat, spacing='proportional', ticks=bounds)
    cb.set_label('Custom cbar')
    ax.set_title('Discrete color mappings')
    plt.show()
