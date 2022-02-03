import os
import shutil
import time

import numpy as np
import torch
from sklearn import metrics
from torch.autograd import Variable
import torch
import torch.nn as nn

import torchvision.models as tmodels
from functools import partial
import collections
import torchvision
import matplotlib.pyplot as plt
from signor.utils.pt.img import img_batch, single_image
from signor.ml.high_dim.dist import pwd_hist
from signor.viz.img.imshow import batch_show
from signor.monitor.probe import summary
from signor.utils.np import tonp
from signor.monitor.stats import stats
import sys


def save_act_across_layers(data=None, verbose = False):
    """ get the activation across different layers.
        https://gist.github.com/Tushar-N/680633ec18f5cb4b47933da7d10902af
    """

    # dummy data_: 10 batches of images with batch size 16
    if data is None:
        dataset = [torch.rand(16, 3, 224, 224) for _ in range(3)]
    else:
        dataset = [data]

    # network: a resnet50
    net = tmodels.resnet50(pretrained=True)

    # a dictionary that keeps saving the activations as they come
    activations = collections.defaultdict(list)

    def save_activation(name, mod, inp, out):
        activations[name].append(out.cpu())

    # Registering hooks for all the Conv2d layers
    # Note: Hooks are called EVERY TIME the module performs a forward pass. For modules that are
    # called repeatedly at different stages of the forward pass (like RELUs), this will save different
    # activations. Editing the forward pass code to save activations is the way to go for these cases.
    for name, m in net.named_modules():
        if type(m) == nn.Conv2d:
            # partial to assign the layer name to each hook
            m.register_forward_hook(partial(save_activation, name))

    # forward pass through the full dataset
    for batch in dataset:
        out = net(batch)

    # concatenate all the outputs we saved to get the the activations for each layer for the whole dataset
    activations = {name: torch.cat(outputs, 0) for name, outputs in activations.items()}

    # just print out the sizes of the saved activations as a sanity check
    if verbose:
        for k, v in activations.items():
            print(k, v.size())
    return activations


def readme():
    pass
    # conv1 torch.Size([15, 64, 112, 112])
    # layer1.0.conv1 torch.Size([15, 64, 56, 56])
    # layer1.0.conv2 torch.Size([15, 64, 56, 56])
    # layer1.0.conv3 torch.Size([15, 256, 56, 56])
    # layer1.0.downsample.0 torch.Size([15, 256, 56, 56])
    # layer1.1.conv1 torch.Size([15, 64, 56, 56])
    # layer1.1.conv2 torch.Size([15, 64, 56, 56])
    # layer1.1.conv3 torch.Size([15, 256, 56, 56])
    # layer1.2.conv1 torch.Size([15, 64, 56, 56])
    # layer1.2.conv2 torch.Size([15, 64, 56, 56])
    # layer1.2.conv3 torch.Size([15, 256, 56, 56])
    # layer2.0.conv1 torch.Size([15, 128, 56, 56])
    # layer2.0.conv2 torch.Size([15, 128, 28, 28])
    # layer2.0.conv3 torch.Size([15, 512, 28, 28])
    # layer2.0.downsample.0 torch.Size([15, 512, 28, 28])
    # layer2.1.conv1 torch.Size([15, 128, 28, 28])
    # layer2.1.conv2 torch.Size([15, 128, 28, 28])
    # layer2.1.conv3 torch.Size([15, 512, 28, 28])
    # layer2.2.conv1 torch.Size([15, 128, 28, 28])
    # layer2.2.conv2 torch.Size([15, 128, 28, 28])
    # layer2.2.conv3 torch.Size([15, 512, 28, 28])
    # layer2.3.conv1 torch.Size([15, 128, 28, 28])
    # layer2.3.conv2 torch.Size([15, 128, 28, 28])
    # layer2.3.conv3 torch.Size([15, 512, 28, 28])
    # layer3.0.conv1 torch.Size([15, 256, 28, 28])
    # layer3.0.conv2 torch.Size([15, 256, 14, 14])
    # layer3.0.conv3 torch.Size([15, 1024, 14, 14])
    # layer3.0.downsample.0 torch.Size([15, 1024, 14, 14])
    # layer3.1.conv1 torch.Size([15, 256, 14, 14])
    # layer3.1.conv2 torch.Size([15, 256, 14, 14])
    # layer3.1.conv3 torch.Size([15, 1024, 14, 14])
    # layer3.2.conv1 torch.Size([15, 256, 14, 14])
    # layer3.2.conv2 torch.Size([15, 256, 14, 14])
    # layer3.2.conv3 torch.Size([15, 1024, 14, 14])
    # layer3.3.conv1 torch.Size([15, 256, 14, 14])
    # layer3.3.conv2 torch.Size([15, 256, 14, 14])
    # layer3.3.conv3 torch.Size([15, 1024, 14, 14])
    # layer3.4.conv1 torch.Size([15, 256, 14, 14])
    # layer3.4.conv2 torch.Size([15, 256, 14, 14])
    # layer3.4.conv3 torch.Size([15, 1024, 14, 14])
    # layer3.5.conv1 torch.Size([15, 256, 14, 14])
    # layer3.5.conv2 torch.Size([15, 256, 14, 14])
    # layer3.5.conv3 torch.Size([15, 1024, 14, 14])
    # layer4.0.conv1 torch.Size([15, 512, 14, 14])
    # layer4.0.conv2 torch.Size([15, 512, 7, 7])
    # layer4.0.conv3 torch.Size([15, 2048, 7, 7])
    # layer4.0.downsample.0 torch.Size([15, 2048, 7, 7])
    # layer4.1.conv1 torch.Size([15, 512, 7, 7])
    # layer4.1.conv2 torch.Size([15, 512, 7, 7])
    # layer4.1.conv3 torch.Size([15, 2048, 7, 7])
    # layer4.2.conv1 torch.Size([15, 512, 7, 7])
    # layer4.2.conv2 torch.Size([15, 512, 7, 7])
    # layer4.2.conv3 torch.Size([15, 2048, 7, 7])

def layers():
    res = ['conv1', 'layer1.0.conv1', 'layer1.0.conv2', 'layer1.0.conv3', 'layer1.0.downsample.0', 'layer1.1.conv1', 'layer1.1.conv2', 'layer1.1.conv3', 'layer1.2.conv1', 'layer1.2.conv2', 'layer1.2.conv3', 'layer2.0.conv1', 'layer2.0.conv2', 'layer2.0.conv3', 'layer2.0.downsample.0', 'layer2.1.conv1', 'layer2.1.conv2', 'layer2.1.conv3', 'layer2.2.conv1', 'layer2.2.conv2', 'layer2.2.conv3', 'layer2.3.conv1', 'layer2.3.conv2', 'layer2.3.conv3', 'layer3.0.conv1', 'layer3.0.conv2', 'layer3.0.conv3', 'layer3.0.downsample.0', 'layer3.1.conv1', 'layer3.1.conv2', 'layer3.1.conv3', 'layer3.2.conv1', 'layer3.2.conv2', 'layer3.2.conv3', 'layer3.3.conv1', 'layer3.3.conv2', 'layer3.3.conv3', 'layer3.4.conv1', 'layer3.4.conv2', 'layer3.4.conv3', 'layer3.5.conv1', 'layer3.5.conv2', 'layer3.5.conv3', 'layer4.0.conv1', 'layer4.0.conv2', 'layer4.0.conv3', 'layer4.0.downsample.0', 'layer4.1.conv1', 'layer4.1.conv2', 'layer4.1.conv3', 'layer4.2.conv1', 'layer4.2.conv2', 'layer4.2.conv3']
    return res

if __name__ == '__main__':
    imgs_ = img_batch(reshape=False)
    imgs = single_image(reshape=True) # img_batch(reshape=True)
    stats(imgs, var_name='imgs')
    summary(imgs)
    imgs = imgs[0:1, :, :, :]  # torch.Size([15, 3, 224, 224])
    grid_img = torchvision.utils.make_grid(imgs, nrow=25)
    plt.imshow(grid_img.permute(1, 2, 0))
    plt.show()

    activations = save_act_across_layers(data=imgs)

    # arr = 255 * np.random.random((16, 7, 7, 3))
    # arr = arr.astype('uint8')
    # batch_show(arr)
    # exit()

    print('-' * 100)

    # for k, v in activations.items():
    #     acts = v  # torch.flatten(v, start_dim=1)
    #     print(k, acts.size())
        # pwd_hist(acts, title=k, show=True)
    # sys.exit()
    for layer in layers():
        acts = activations[layer].detach()  # torch.Size([1, 64, 56, 56])
        acts = acts.permute(1, 0, 2, 3) # torch.Size([64, 1, 56, 56])

        # acts = acts.permute(1,2,3,0)
        summary(acts)
        grid_img = torchvision.utils.make_grid(acts, nrow=int(np.sqrt(acts.size(0))))
        plt.imshow(grid_img.permute(1, 2, 0))
        plt.title(f'act at {layer}')
        plt.show()
    sys.exit()

    acts = acts.permute(0, 2, 3, 1)
    acts = tonp(acts[:64, :, :, :]).astype('uint8')
    stats(acts)
    summary(acts, 'acts')
    batch_show(acts, grid_desc=('4v', '4h'), nr_show=64)
