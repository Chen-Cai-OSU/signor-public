""" visualize across layers """

import torch
import torch.nn as nn

import torchvision.models as tmodels
from functools import partial
import collections
from signor.ioio.dir import torch_io
from signor.monitor.time import timefunc
from signor.monitor.probe import summary
from signor.configs.util import subset_dict, dict2name
from signor.utils.naming import param2name
from torch.utils.data import DataLoader
from signor.utils.pt.shape import dataloader_shape

def save_activation(activations, name, mod, inp, out):
    activations[name].append(out.cpu())

@timefunc
def save_act_across_layers(net=None, dataloader=None, save=False, dev='cuda', verbose = True, **save_kwargs):
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
        net = tmodels.resnet50(pretrained=True)
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
        if True: # type(m) == nn.Conv2d:# 'encoder': #True:  # type(m) == nn.Conv2d: # todo: add pattern
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
    if verbose:
        for name, outputs in activations.items():
            print(name, summary(outputs))

    activations = {name: torch.cat(outputs, 0) for name, outputs in activations.items()}
    targets = torch.cat(targets, dim=0)


    # just print out the sizes of the saved activations as a sanity check
    for layer, v in activations.items():
        print('{:20}'.format(layer), '\t\t', v.size())

    if save:  # todo: add pattern for higher abstraction
        for layer, v in activations.items():
            if len(str(layer)) < 2: continue
            tio = torch_io(verbose=verbose, dir=save_kwargs.get('dir', None))

            # epoch = save_kwargs.get('epoch', 0)
            # batch_ = save_kwargs.get('batch', 0)
            # name = '_'.join(['epoch', str(epoch), 'batch', str(batch_), str(layer), ''])[:-1] #todo: better name
            save_kwargs = subset_dict(save_kwargs).exclude(['dir'])
            name = dict2name(save_kwargs)

            tio.dump([v, targets], name)
            print()
        del activations, targets, out
        return
    else:
        return activations


if __name__ == '__main__':
    from signor.utils.pt.shape import fake_Dataset

    x = [torch.rand(3, 224, 224)] * 100
    y = [1] * 100
    dataloader = DataLoader(fake_Dataset(x, y), batch_size=10)
    dataloader_shape(dataloader)

    save_kwargs ={'model': 'resnet50'}
    acts = save_act_across_layers(dataloader=dataloader, save=True, net=None, **save_kwargs)
