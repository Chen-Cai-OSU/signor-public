import torch.nn.functional as F
from torch import nn

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import GridSearchCV
from skorch import NeuralNetClassifier

from signor.datasets.image_classification.mnist import load_mnist
from signor.monitor.probe import summary
import torch

from signor.utils.pt.feat_extractor import save_act_across_layers

torch.manual_seed(0)
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, BatchNorm1d as BN


class MLP(nn.Module):

    def __init__(self, h_sizes):
        """
        https://bit.ly/3drTQ6C
        :param h_sizes: a list of num of hidden units
        :param out_size: output num of classes
        """
        super(MLP, self).__init__()
        self.hidden = nn.ModuleList()
        for k in range(len(h_sizes)-1):
            layer = Seq(nn.Linear(h_sizes[k], h_sizes[k+1]), ReLU(), BN(h_sizes[k+1]))
            self.hidden.append(layer)

    def forward(self, x):
        for layer in self.hidden:
            x = layer(x)
        x = F.softmax(x, dim=-1)

        return x


class ClassifierModule(nn.Module):
    def __init__(
            self,
            num_units = [784, 100, 10],
            dropout=0.5,
    ):

        input_dim, hidden_dim, output_dim = num_units
        super(ClassifierModule, self).__init__()
        self.dropout = nn.Dropout(dropout)

        self.hidden = nn.Linear(input_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, output_dim)

    def forward(self, X, **kwargs):
        X = F.relu(self.hidden(X))
        X = self.dropout(X)
        X = F.softmax(self.output(X), dim=-1)
        return X


class MyModule(nn.Module):
    def __init__(self, num_units=10, nonlin=F.relu):
        super(MyModule, self).__init__()

        self.dense0 = nn.Linear(784, num_units)
        self.nonlin = nonlin
        self.dropout = nn.Dropout(0.5)
        self.dense1 = nn.Linear(num_units, 10)
        self.output = nn.Linear(10, 10)

    def forward(self, X, **kwargs):
        X = self.nonlin(self.dense0(X))
        X = self.dropout(X)
        X = F.relu(self.dense1(X))
        X = F.softmax(self.output(X), dim=-1)
        return X



if __name__ == '__main__':
    train_data, _, _ = load_mnist(data_dir='/home/cai.507/Documents/DeepLearning/Signor/data/')
    X, y = train_data

    # X, y = make_classification(1000, 20, n_informative=10, random_state=0)
    X = X.astype(np.float32)
    y = y.astype(np.int64)

    net = NeuralNetClassifier(
        MLP,
        max_epochs=20,
        lr=0.1,
        device='cuda:1',
        batch_size=256,
    )

    summary(X, 'X')
    acts = save_act_across_layers(net=net.module([784, 10, 20]), data=torch.tensor(X))
    summary(acts)
    exit()

    print(list(net.module([10, 20]).named_modules()))
    exit()
    # net = NeuralNetClassifier(
    #     ClassifierModule,
    #     max_epochs=20,
    #     lr=0.1,
    #     device='cuda:1',
    #     batch_size=256,
    # )

    # net = NeuralNetClassifier(
    #     MyModule,
    #     max_epochs=10,
    #     device='cuda:1',
    #     batch_size=128,
    #     lr=0.1,
    #     iterator_train__shuffle=True,
    # )

    params = {
        'lr': [0.01, ],
        'max_epochs': [30],
        'module__h_sizes': [[784, 100, 10]],
    }

    gs = GridSearchCV(net, params, refit=False, cv=5, scoring='accuracy')
    gs.fit(X, y)
    print(gs.best_score_, gs.best_params_)