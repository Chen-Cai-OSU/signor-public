# Created at 2020-04-13
# Summary: extractor feature at certain layers of emb https://github.com/polarisZhao/pytorch-cookbook
import collections
from functools import partial
from pprint import pprint

import skorch
import torch
import torchvision

from signor.format.format import banner
from signor.monitor.probe import summary
from signor.utils.pt.summary import gnn_model_summary
from signor.utils.pt.util import model_device


class FeatureExtractor(torch.nn.Module):
    """Helper class to extract several convolution features from the given
    pre-trained model.
    Attributes:
    _model, torch.nn.Module.
    _layers_to_extract, list<str> or set<str>
    Example:
    
    """

    def __init__(self, pretrained_model, layers_to_extract):
        torch.nn.Module.__init__(self)
        self._model = pretrained_model
        self._model.eval()
        self._layers_to_extract = set(layers_to_extract)

    def forward(self, x):
        with torch.no_grad():
            conv_representation = []
            for name, layer in self._model.named_children():
                x = layer(x)
                if name in self._layers_to_extract:
                    conv_representation.append(x)
        return conv_representation

def save_act_across_layers(net, data=None, verbose = False):
    """ get the activation across different layers.
        https://gist.github.com/Tushar-N/680633ec18f5cb4b47933da7d10902af
    """

    # dummy data_: 10 batches of images with batch size 16

    assert isinstance(net, (skorch.classifier.NeuralNet, torch.nn.Module)),  f'net is of type {type(net)}.'
    device = model_device(net)

    if data is None:
        dataset = [torch.rand(16, 3, 224, 224) for _ in range(3)]
    else:
        if isinstance(data, list):
            dataset = data
        else:
            dataset = [data]

    # a dictionary that keeps saving the activations as they come
    activations = collections.defaultdict(list)

    def save_activation(name, mod, inp, out):
        activations[name].append(out.cpu())

    # Registering hooks for all the Conv2d layers
    # Note: Hooks are called EVERY TIME the module performs a forward pass. For modules that are
    # called repeatedly at different stages of the forward pass (like RELUs), this will save different
    # activations. Editing the forward pass code to save activations is the way to go for these cases.
    if isinstance(net, torch.nn.Module):
        modules = net.named_modules()
    else:
        raise NotImplementedError

    for name, m in modules:
        m.register_forward_hook(partial(save_activation, name))

    # forward pass through the full dataset
    for batch in dataset:
        batch = batch.to(device)
        out = net(batch)

    # concatenate all the outputs we saved to get the the activations for each layer for the whole dataset
    activations = {name: torch.cat(outputs, 0) for name, outputs in activations.items()}

    # just print out the sizes of the saved activations as a sanity check
    if verbose:
        for k, v in activations.items():
            print(k, v.size())
            summary(v, k)
    return activations


if __name__ == '__main__':
    model = torchvision.models.resnet152(pretrained=True)
    for name, layer in model.named_children():
        banner(name)
        print(layer)

    # pprint(list(model.named_children()))
    summary(list(model.named_children()), 'model.named_children()')

    exit()
    model = torch.nn.Sequential(collections.OrderedDict(list(model.named_children())[:-1]))
    gnn_model_summary(model)

    image = torch.rand(16, 3, 224, 224)
    conv_representation = FeatureExtractor(
        pretrained_model=model,
        layers_to_extract={'layer1', 'layer2', 'layer3', 'layer4'})(image)

    summary(conv_representation, 'conv_representation')
