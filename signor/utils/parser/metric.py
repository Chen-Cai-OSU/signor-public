# Created at 2021-02-22
# Summary: parse training log

# handle other metrics

from collections import defaultdict

import pandas as pd

from signor.format.format import red


def get_metric(content, lastk, end=None, aggreator='mean'):
    """
    :param content: a list of lines, each line is like
           Epoch: 000, Train Loss: 0.333, Test RMSEN loss: 0.333/0.1657
    :param lastk: non-negative int
    :return:
    """
    assert lastk >= 0
    assert aggreator in ['mean', 'min', None]
    content = [line for line in content if line.__contains__('Train')]

    if len(content) < lastk: # todo: better handling
        print(red(f'Length of content({len(content)}) < {lastk}'))
        return {} # dict(zip(self.keys, self.defaultvals))

    if end is None:
        content = content[-lastk:]
    else:
        content = content[end-lastk: end]

    ret = defaultdict(list)
    for line in content:
        if 'Epoch' not in line: continue

        idx = line.index('Epoch')
        line = line[idx:]
        tmp = line.split(', ')
        # ['Epoch 90: train loss: 0.0025', 'test loss: 0.0', 'cart loss: 0.0014/0.3154', 'time: 0', '\n']

        epoch = tmp[0].split(':')[1].strip(' ')
        train_loss = tmp[1].split(':')[1].strip(' ')
        test_loss = tmp[2].split(':')[1].strip(' ')
        test_cart_loss, trivial_loss = test_loss.split('/')
        # time = tmp[3].split(':')[1].strip(' ')
        args = epoch, train_loss, test_cart_loss, trivial_loss
        epoch, train_loss, test_cart_loss, trivial_loss = list(map(float, args))
        for k in ['epoch', 'train_loss', 'test_cart_loss', 'trivial_loss']:
            ret[k].append(eval(k))

    df = pd.DataFrame(ret)
    if aggreator == 'mean':
        perf_dict = dict(df.mean(axis=0))
    elif aggreator == 'min':
        perf_dict = dict(df.min(axis=0))
    elif aggreator == None:
        perf_dict = df

    return perf_dict
