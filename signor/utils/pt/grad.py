#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : grad.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 05/08/2018
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

import functools
import matplotlib.pyplot as plt
import torch
import numpy as np
from matplotlib.lines import Line2D
from pprint import pprint
from signor.format.format import delimiter

__all__ = ['no_grad_func']


def no_grad_func(func):
    @functools.wraps(func)
    def new_func(*args, **kwargs):
        with torch.no_grad():
            return func(*args, **kwargs)

    return new_func


class gradient_check():
    # https://github.com/alwynmathew/gradflow-check
    def __init__(self, model):
        self.model = model

    def check_shape(self, shape = False):
        if shape:
            print('The size of all gradient across layers')
            for k, v in self.model.named_parameters():
                assert v.grad is not None, 'grad is None.'
                print('{:20}'.format(k), '\t\t', v.grad.size())
        else:
            print('The gradient across layers')
            for k, v in self.model.named_parameters():
                print('{:20}'.format(k), '\t\t', v.grad)

        delimiter().large()



def plot_grad_flow(named_parameters, show=False, text = False):
    ave_grads = []
    layers = []
    for n, p in named_parameters:
        if (p.requires_grad) and ("bias" not in n):
            layers.append(n)
            assert p.grad is not None, 'grad is None.'
            ave_grads.append(p.grad.abs().mean())

    if text:
        print('gradient absolute average across layers (bias excluded)')
        assert len(layers) == len(ave_grads)
        for i in range(len(layers)):
            layer, ave_grad = layers[i], ave_grads[i]
            print('{:20}'.format(layer), '\t\t', ave_grad)
        delimiter().small()
        return

    plt.plot(ave_grads, alpha=0.3, color="b")
    plt.hlines(0, 0, len(ave_grads) + 1, linewidth=1, color="k")
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(xmin=0, xmax=len(ave_grads))
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)

    if show:
        plt.show()



def plot_grad_flow_v2(named_parameters):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.

    Usage: Plug this function in Trainer class after loss.backwards() as
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''

    ave_grads = []
    max_grads = []
    layers = []
    for n, p in named_parameters:
        if (p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())

    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads) + 1, lw=2, color="k")
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom=-0.001, top=0.02)  # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])


if __name__ == '__main__':
    import torchvision.models as tmodels

    net = tmodels.resnet18(pretrained=True)

    gradient_check(net).check_shape(shape=False)
    exit()

    plot_grad_flow(net.named_parameters())
    exit()

    plot_grad_flow_v2(net.named_parameters())

