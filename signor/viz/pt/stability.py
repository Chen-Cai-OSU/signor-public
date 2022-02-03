""" noise stability of layers """

import collections
from functools import partial

import torch
import torch.nn as nn
import torchvision.models as tmodels
from torch.utils.data import DataLoader

from signor.format.format import pf
from signor.monitor.probe import summary
from signor.monitor.time import timefunc
from signor.utils.np import norm
from signor.utils.pt.shape import fake_Dataset


def save_activation(activations, name, mod, inp, out):
    activations[name].append(out.cpu())


@timefunc
def save_act_across_layers(net=None, dataloader=None, dev='cuda'):
    """
        get the activation across different layers.
        https://gist.github.com/Tushar-N/680633ec18f5cb4b47933da7d10902af
        a dict keyed by different layers and values are tensors of different shape
        # todo: memory issue. gpu memory increase along training epoch
    :param net:  model
    :param dataloader: one batch
    :param save: save to file or not
    :param dev: device
    :param save_kwargs: {dir: xxx, epoch: 1}
    :return:
    """

    # dummy data_: 3 batches of images with batch size 16
    if dataloader is None:
        dataset = [torch.rand(16, 3, 224, 224) for _ in range(3)]
    else:
        dataset = dataloader

    # network: a resnet50
    if net is None:
        net = tmodels.resnet50(pretrained=True) # tmodels.vgg19(pretrained=True)
    else:
        net.eval()
    net.to(dev)

    # a dictionary that keeps saving the activations as they come
    activations = collections.defaultdict(list)
    targets = []

    # Registering hooks for all the Conv2d layers
    # Note: Hooks are called EVERY TIME the module performs a forward pass. For modules that are
    # called repeatedly at different stages of the forward pass (like RELUs), this will save different
    # activations. Editing the forward pass code to save activations is the way to go for these cases.

    for name, m in net.named_modules():
        if type(m) == nn.Conv2d:  # 'encoder': #True:  # type(m) == nn.Conv2d: # todo: add pattern
            # partial to assign the layer name to each hook
            m.register_forward_hook(partial(save_activation, activations, name))

    for data in dataset:
        if isinstance(data, (list, tuple)):
            assert len(data) == 2
            data, target = data
            data = data.to(dev)
        out = net(data)
        targets.append(target)

    # concatenate all the outputs we saved to get the the activations for each layer for the whole dataset
    for name, outputs in activations.items():
        print(name, summary(outputs))

    activations = {name: torch.cat(outputs, 0) for name, outputs in activations.items()}
    targets = torch.cat(targets, dim=0)

    # just print out the sizes of the saved activations as a sanity check
    for layer, v in activations.items():
        print('{:20}'.format(layer), '\t\t', v.size())

    return activations

def relative_error(a, b):
    return 100 * (b-a)/float(a)

if __name__ == '__main__':
    # todo: add real dataset
    x = [torch.rand(3, 224, 224)] * 100
    x_ = [ d + torch.rand(3, 224, 224) * 1 for d in x]
    y = [1] * 100

    dl = DataLoader(fake_Dataset(x, y), batch_size=50)
    dl_noise = DataLoader(fake_Dataset(x_, y), batch_size=50)

    acts = save_act_across_layers(dataloader=dl, net=None, dev='cpu')
    acts_ = save_act_across_layers(dataloader=dl_noise, net=None, dev='cpu')

    for k in acts.keys():
        v1, v2 = acts[k], acts_[k]
        print(f'layer: {k}')
        print(f'norm: {pf(norm(v1))}/{pf(norm(v2))}')
        print(f'relative error (percent): {pf(relative_error(norm(v1), norm(v2)))}')
        print()
