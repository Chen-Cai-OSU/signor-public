#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : naming.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 01/18/2017
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

from collections import OrderedDict

__all__ = ['class_name', 'func_name', 'method_name', 'class_name_of_method']


def class_name(instance_or_class):
    if isinstance(instance_or_class, type):
        return func_name(instance_or_class)
    return func_name(instance_or_class.__class__)


def func_name(func):
    return func.__module__ + '.' + func.__qualname__


def method_name(method):
    assert '.' in method.__qualname__, '"{}" is not a method.'.format(repr(method))
    return func_name(method)


def class_name_of_method(method):
    name = method_name(method)
    return name[:name.rfind('.')]


def param2name(d):
    assert isinstance(d, dict)
    d = OrderedDict(sorted(d.items()))
    name = ''
    for k, v in d.items():
        name += f'{k}_{v}_'
    return name[:-1]



if __name__ == '__main__':
    d1 = {'a': 1, 'c': 3, 'b': 2}
    d2 = {'a': 1, 'b': 2, 'c': 3}

    print(param2name(d1))
    print(param2name(d2))
    exit()

    x = 10
    print(class_name(x))

    import numpy as np

    x = np.random.random((10, 3))
    print(class_name(x))
