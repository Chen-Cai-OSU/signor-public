""" latent space visualization """

import matplotlib.pyplot as plt
import torch
import numpy as np

from signor.monitor.probe import summary
from signor.utils.np import tonp
from signor.viz.color import set_color
from signor.viz.dim_red import dim_reducer

fig, ax = plt.subplots(3, 3, figsize=(6, 6))
for epoch in np.linspace(0, 80, 9): #[1, 10, 20, 30, 40]:  # [1, 10, 20]:
    epoch = int(epoch)
    i, j = int((epoch/10) // 3), int((epoch/10) % 3)
    print(epoch, i, j)
    # continue

    f = f'/tmp/epoch_{epoch}_batch_468_encoder'
    d = dim_reducer(dir='/tmp/')
    x, target = torch.load(f)
    x, target = tonp(x), tonp(target)
    x = x.reshape((60000, -1))
    n_sample = 1000

    summary(x, 'x')
    summary(target, 'target')

    x, target = d.sampler(data=(x, target), s=n_sample, verbose=True)

    emb = d.pca(x)
    summary(target, 'sampled target:')

    target, cmap, norm, bounds = set_color(target)

    scat = ax[i][j].scatter(emb[:, 0], emb[:, 1], c=target, cmap=cmap, s=3, norm=norm)
    # create the colorbar
    # cb = plt.colorbar(scat, spacing='proportional', ticks=bounds)
    # cb.set_label('Labels')
    ax[i][j].set_title(f'pca of {n_sample} points. Epoch {epoch}. ', fontsize=5)
    ax[i][j].axis('off')
    print()

plt.show()

    # d.show_emb(emb, title=f'pca. epoch:{epoch}')
