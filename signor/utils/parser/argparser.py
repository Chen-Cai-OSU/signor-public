# Created at 2021-02-28
# Summary: common argparse for ML


import argparse

def defalut_argparser():
    parser = argparse.ArgumentParser(description='gnn baseline')
    # optim
    parser.add_argument('--device', type=str, default='cuda:0', help='')
    parser.add_argument('--n_epoch', type=int, default=200, help='')
    parser.add_argument('--depth', type=int, default=3, help='')
    parser.add_argument('--n_graph', type=int, default=300, help='number of graph')
    parser.add_argument('-l', '--loss', type=str, default='RMSEN', help='type of loss',
                        choices=['MSE', 'RMSE', 'RMSEN'])

    return parser




