""" viz pygeo data
    Summary of torch_geometric.data.data.Data:
      edge_attr                LongTensor          [56, 1]         16.5(mean)  13.0(min)  22.0(max)  17.5(median)  2.54(std)  10.0(unique)
      edge_index               LongTensor          [2, 56]         4.24(mean)   0.0(min)   9.0(max)   4.0(median)  2.94(std)  10.0(unique)
      go_target_downstream     DoubleTensor        [7]              0.5(mean)   0.0(min)  0.86(max)  0.57(median)  0.26(std)   7.0(unique)
      x                        LongTensor          [10, 2]         31.4(mean)   1.0(min)  82.0(max)  23.0(median) 32.66(std)   4.0(unique)
"""
import torch_geometric
from torch_geometric.data import Data

import matplotlib.pyplot as plt
import numpy as np

from signor.ml.high_dim.dist import _hist_show
from signor.monitor.probe import summary
from signor.utils.np import tonp


# from signor.viz.matrix import matshow

class pygeo_viz():
    """ visuzliae torch_geometric.data.data.Data """
    def __init__(self, dataset, sample_size = None):
        assert isinstance(dataset[0], torch_geometric.data.data.Data)
        n = len(dataset)
        self.n = n
        if sample_size is None:
            random_indices = list(range(n))
        else:
            random_indices = np.random.randint(0, n, size=min(sample_size, n)).tolist()
        self.filter_data_list = [dataset[i] for i in random_indices]


    def get_attr(self, attr):
        """
        extract all data for attribute attr
        :param attr: for example, go_target_downstream
        :return: np.array
        """
        x = [tonp(data.__getattribute__(attr)) for data in self.filter_data_list]
        if x[0].ndim == 1:
            x = [x_.reshape((1, len(x_))) for x_ in x]
        x = np.concatenate(x, axis=0)
        return x


    def viz_mat(self, attr):
        x = self.get_attr(attr)
        self.matshow(x, var_name=attr)

    def viz_slice_mat(self, attr, index = 0):
        """ visualize different feats """
        x = self.get_attr(attr)
        assert x.ndim == 2, f'Expect x ndim to be 2 but got x {summary(x)}'
        x = x[:, index]
        summary(x)
        _hist_show(x, f'slice of {attr} at index {index}')

    @staticmethod
    def matshow(x, var_name='x'):
        """ improved matshow for better height/width ratio """
        assert isinstance(x, np.ndarray)
        summary(x)
        plt.imshow(x)
        # if x.shape[0] > x.shape[1] * 10:
        plt.axes().set_aspect('auto')  # http://bit.ly/37V4xL0
        plt.colorbar()
        plt.title(f'{var_name} {np.shape(x)}')
        plt.show()


if __name__ == '__main__':
    dataset = [Data(x=np.random.random((128))) for _ in range(1000)]
    pv = pygeo_viz(dataset, sample_size=1000)
    pv.viz_mat('x')
    pv.viz_slice_mat('x', index=0)






