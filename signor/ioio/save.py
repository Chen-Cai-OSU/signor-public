import os
import pickle
from functools import partial

import numpy as np
import torch

from signor.monitor.stats import stats
from signor.monitor.time import timefunc
import json

tf = partial(timefunc, threshold=0)


class save():
    def __init__(self, data= None):
        self.data = data
    
    @tf    
    def pickle_save(self, f):
        assert self.data is not None, f'Data is None'
        with open(f, 'wb') as f_:
            pickle.dump(self.data, f_)
    
    @tf
    def pickle_load(self, f):
        assert os.path.isfile(f), f'File {f} does not exist.'
        with open(f, 'rb') as f_:
            data = pickle.load(f_)
        return data

    @tf
    def torch_save(self, f):
        torch.save(self.data, f)

    @tf
    def torch_load(self, f):
        assert os.path.isfile(f)
        return torch.load(f)

    # todo: not success
    @tf
    def json_save(self, f):
        with open(f, 'w') as f:
            json.dump(self.data, f)
    @tf
    def json_load(self, f):
        with open(f, 'r') as f:
            data = json.load(f)
        return data


def pickle2json(f):
    assert 'pickle' == f[-6:]
    s = save()
    data = s.pickle_load(f)
    s.data = data
    newf = f[:-6] + 'json'
    print(newf)
    s.json_save(newf)
    json_data = s.json_load(newf)
    assert data == json_data


if __name__ == '__main__':
    f = '/Users/checai/Documents/fashion-space-clustering/data/walid/pickle/asin_to_teen_asin_clean.pickle'
    save().json_load(f)
    # pickle2json(f)
    exit()
    data = np.random.random((10000, 10000))

    # from signor.graph.cgcnn.code.data_util import dummy_g
    # data = dummy_g()

    f = '/tmp/tmp.VTq151reqL'
    for method in ['pickle', 'torch']:
        s = save(data=data) # torch_save(f) # pickle_save(f)
        getattr(s, method + '_save')(f)
        data_load = getattr(s, method + '_load')(f) # save().torch_load(f) # pickle_load(f)

        # print(data)
        # print(data_load)
        stats(data-data_load)
        assert (data == data_load).all()