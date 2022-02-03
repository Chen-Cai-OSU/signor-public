import torch
import torch.nn.functional as F

from signor.utils.pt.grad import no_grad_func
from signor.utils.pt.meta import as_float

__all__ = [
    'binary_classification_accuracy',
]


@no_grad_func
def binary_classification_accuracy(pred, label, name='', saturation=True):
    if name != '':
        name = '/' + name
    prefix = 'accuracy' + name
    pred = pred.view(-1)  # Binary accuracy
    label = label.view(-1)
    acc = label.float().eq((pred > 0.5).float())
    if saturation:
        sat = 1 - (pred - (pred > 0.5).float()).abs()
        return {
            prefix: as_float(acc.float().mean()),
            prefix + '/saturation/mean': as_float(sat.mean()),
            prefix + '/saturation/min': as_float(sat.min())
        }
    return {prefix: as_float(acc.float().mean())}

# todo: import bug
if __name__ == '__main__':
    from signor.utils.pt.random_ import random_binary

    pred = random_binary(size=(10, 1), ratio=0.5)
    label = random_binary(size=(10, 1), ratio=0.5)
    binary_classification_accuracy(pred, label)
