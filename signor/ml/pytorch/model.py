from signor.utils.random_ import fix_seed

# fix_seed()
import numpy as np

from signor.format.format import red
from signor.monitor.probe import summary


def num_trainable_params(model, verbose=False):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    n_params = sum([np.prod(p.size()) for p in model_parameters])

    if verbose:
        # print(f'Model: {model}')
        model_parameters = filter(lambda p: p.requires_grad, model.parameters())
        for i, param in enumerate(model_parameters):
            # print(param)
            summary(param, f'{i}-model-param')
            # import pdb; pdb.set_trace()
    print(f'model has {red(n_params)} trainable params\n')
    return n_params


if __name__ == '__main__':
    import torchvision.models as tmodels

    model = tmodels.resnet18(pretrained=True)
    num_trainable_params(model, verbose=True)
