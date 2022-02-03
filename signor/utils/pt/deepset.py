# Created at 2020-04-07
# Summary: implement a generic deepset. modified from https://github.com/manzilzaheer/DeepSets/blob/master/PopStats/model.py
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from signor.monitor.probe import summary
from signor.utils.dict import filter_dict
from signor.utils.pt.feat_extractor import FeatureExtractor, save_act_across_layers
from signor.utils.pt.random_ import fix_seed

fix_seed()
kwargs = {'extractor_hidden_dim': [50, 100],
          'regressor_hidden_dim': [100, 100, 50],
          'extractor_nl': nn.ELU,
          'regressor_nl': nn.ELU
          }


class DeepSet(nn.Module):

    def __init__(self, in_features=94, set_features=50, out_dim = 1, task='reg', bn=False,  **kwargs):
        """

        :param in_features:
        :param set_features:
        :param bn:
        :param weight: weights for each element in set
        :param kwargs:
        """
        super(DeepSet, self).__init__()

        print(f'DeepSet kw: {kwargs}')
        extractor_hidden_dim = kwargs['extractor_hidden_dim']
        regressor_hidden_dim = kwargs['regressor_hidden_dim']
        extractor_nl = kwargs['extractor_nl']
        regressor_nl = kwargs['regressor_nl']

        self.in_features = in_features
        self.out_features = set_features
        self.bn = bn
        self.task = task

        self.feature_extractor = self.mlp(in_features, set_features, extractor_hidden_dim, nl=extractor_nl, bn=False, last_nn=False)

        self.agg = partial(torch.sum, dim=1)  # torch.sum(dim=1)
        self.regressor = self.mlp(set_features, out_dim, regressor_hidden_dim, nl=regressor_nl, bn=self.bn, bn_dim=1)

        self.add_module('feature_extractor', self.feature_extractor)
        self.add_module('regressor', self.regressor)

    def mlp(self, input_dim, output_dim, hidden, nl=nn.ELU, bn=False, bn_dim = 1, last_nn = False):
        assert isinstance(hidden, list)

        hidden = [input_dim] + hidden + [output_dim]
        n = len(hidden)
        modules = []
        for i in range(n - 2):
            inner_module = []
            inner_module.append(nn.Linear(hidden[i], hidden[i + 1]))
            inner_module.append(nl(inplace=True))
            if bn:
                if bn_dim == 1:
                    inner_module.append(nn.BatchNorm1d(hidden[i + 1]))
                elif bn_dim == 2:
                    inner_module.append(nn.BatchNorm2d(hidden[i + 1]))
                elif bn_dim == 3:
                    inner_module.append(nn.BatchNorm3d(hidden[i + 1]))
                else:
                    raise NotImplementedError
            modules.append(nn.Sequential(*inner_module))

        modules.append(nn.Linear(hidden[n - 2], hidden[n - 1]))
        if last_nn:
            modules.append(nn.ELU(inplace=True))
        return nn.Sequential(*modules)

    def reset_parameters(self):
        for module in self.children():
            reset_op = getattr(module, "reset_parameters", None)
            if callable(reset_op):
                reset_op()

    def forward(self, input):
        x = input
        x = self.feature_extractor(x)
        x = self.agg(x)  #
        x = self.regressor(x)
        if self.task == 'clf':
            x = F.softmax(x, dim=-1)
        return x

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'Feature Exctractor=' + str(self.feature_extractor) \
               + '\n Set Feature' + str(self.regressor) + ')'


def remove_sequential(network):
    all_layers = []
    for layer in network.children():
        if isinstance(layer, nn.Sequential) : # if sequential layer, apply recursively to layers in sequential layer
            all_layers += remove_sequential(layer)
        if list(layer.children()) == []: # if leaf node, add it to list
            all_layers.append(layer)
    return all_layers



if __name__ == '__main__':
    x = [torch.rand(20, 4, 30)] * 2
    model = DeepSet(in_features=30, set_features=10, **kwargs)
    activations = save_act_across_layers(model, verbose=True, data=x)
    layers = filter_dict(activations, keys=['feature_extractor', 'regressor.5'])
    summary(layers)
    exit()


    representation = FeatureExtractor(
        pretrained_model=model,
        layers_to_extract={'feature_extractor', 'regressor'})(x)
    summary(representation, 'representation')

    ds = DeepSet(in_features=30, set_features=10, **kwargs)
    out = ds(x)
    summary(out)
