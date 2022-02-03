# Created at 2021-02-22
# Summary: used for test tuna

def main():
    import os
    import os.path as osp
    from collections import defaultdict
    from pprint import pprint
    from time import time

    import numpy as np
    import pandas as pd
    from tqdm import tqdm

    t0 = time()
    from signor.configs.util import dict2name, dict_product, load_configs
    from signor.format.format import pf, banner, red
    from signor.ioio.dir import sig_dir
    from signor.utils.dict import merge_two_dicts
    from signor.utils.str import filter_logtime
    from signor.utils.random_ import fix_seed2
    fix_seed2()

    import argparse
    from collections import namedtuple
    from functools import partial

    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    from signor.graph.permeability.share.evaluation import loss_selector, bl, nick_path, write_bl
    from signor.graph.permeability.share.split import Splitter
    from signor.graph.permeability.util.image_util import Dataset
    from signor.ml.pytorch.model import num_trainable_params
    from signor.monitor.probe import summary
    from signor.monitor.time import timefunc
    from signor.utils.notify import slack
    from signor.utils.parallel import torch_parallel
    from copy import deepcopy

    from signor.graph.permeability.model.gin_edge import GNN_graphpred
    from signor.graph.permeability.share.split import Splitter
    from signor.graph.permeability.util.data import PyGDataset
    from signor.graph.permeability.util.model_util import train, test, get_path
    from signor.ml.classic.normalizer import Normalizer
    from signor.monitor.probe import summary

    import torch

    from torch_geometric.data import DataLoader

    import argparse
    from signor.utils.random_ import fix_seed2
    fix_seed2()

    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    # The class GatedBlock inherit from the class torch.nn.Module.
    # It contains one convolution, some ReLU and multiplications

    from signor.graph.permeability.model.cnn import train, test
    from signor.graph.permeability.share.split import Splitter
    from signor.graph.permeability.util.image_util import Dataset
    from signor.ml.pytorch.model import num_trainable_params
    from signor.monitor.probe import summary
    from signor.utils.notify import slack
    from signor.utils.parallel import torch_parallel
    from torch.utils.data import DataLoader
    from tqdm import tqdm

def main1():
    import signor

if __name__ == '__main__':
    # profile with tuna
    # python -mcProfile -o program.prof utils/import-test.py && tuna program.prof
    main1()
