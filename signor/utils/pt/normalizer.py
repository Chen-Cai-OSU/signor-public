""" a normalizer for pytorch geometric """
import torch
from torch_geometric.data.data import Data

import numpy as np

from signor.monitor.probe import summary
from signor.utils.pt.shape import fake_Dataset
from signor.viz.pt.pygeo import pygeo_viz
from signor.ml.classic.normalizer import Multinormalizer

SAMPLE_SIZE = 1000
DIM = 100

class _data(fake_Dataset):
    def __init__(self, *args):
        super(_data, self).__init__(*args)

    def __getitem__(self, index):
        X = self.x[index]  # pt.load('data_/' + ID + '.pt')
        y = self.y[index]
        return Data(pos=X, y=y)

    def reset_y(self, i, y):
        assert isinstance(y, torch.Tensor)
        self.y[i] = y


if __name__ == '__main__':
    x = [torch.rand(SAMPLE_SIZE, DIM) for _ in range(1000)]
    y = [torch.Tensor([10 * np.random.random((5))]).type(torch.FloatTensor) for _ in range(1000)]

    train_dataset = _data(x, y)
    test_dataset = _data(x, y)
    summary(train_dataset[0])
    y = pygeo_viz(train_dataset, sample_size=200).get_attr('y')


    n = Multinormalizer(y)

    yy = [[10 * np.random.random((5))] for _ in range(1000)]
    yy = np.concatenate(yy, axis=0)
    yy_norm = n.norm(yy)
    yy_denorm = n.denorm(yy_norm)

    # reset train_dataset.y

    for i in range(len(train_dataset)):
        summary(train_dataset[i].y, name='before')
        train_dataset.reset_y(i, yy_norm[[i], :])
        summary(train_dataset[i].y, name='after')
        exit()
        # assert (train_dataset[i].y == yy_denorm[i]).all()
    summary(train_dataset[1])
    exit()

    newy = pygeo_viz(train_dataset, sample_size=None).get_attr('y')
    summary(yy_denorm, 'yy_denorm')
    summary(torch.tensor(newy), 'newy')
    # summary(yy_denorm - torch.tensor(newy), 'diff')

    exit()

    summary(yy_norm, 'y_norm')
    for i in range(5):
        summary(yy_norm[:, i], f'y_norm[:, {i}]')


    summary(yy_denorm, 'yy_denorm')
    summary(torch.tensor(yy), 'yy')