""" model summary """
import torch
import os.path as osp

import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import GATConv

from signor.format.format import pf
# torch.set_printoptions(precision=4)
import numpy as np

def gnn_model_summary(model):
    model_params_list = list(model.named_parameters())
    print("-----------------------------------------------------------------------------------")
    line_new = "{:>30}  {:>25} {:>15}".format("Layer.Parameter", "Param Tensor Shape", "Param #")
    print(line_new)
    print("-----------------------------------------------------------------------------------")
    for elem in model_params_list:
        p_name = elem[0]
        p_shape = list(elem[1].size())
        param = elem[1].T
        p_count = torch.tensor(elem[1].size()).prod().item()
        line_new = "{:>35}  {:>25} {:>15} ".format(p_name, str(p_shape), str(p_count))
        print(line_new)
    print("-----------------------------------------------------------------------------------")
    total_params = sum([param.nelement() for param in model.parameters()])
    print("Total params:", total_params)
    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Trainable params:", num_trainable_params)
    print("Non-trainable params:", total_params - num_trainable_params)

class model_param(object):
    def __init__(self, model):
        self.model = model
        self.model_params_list = list(model.named_parameters())

    def mean(self):
        return np.mean([torch.mean(elem[1].T).item() for elem in self.model_params_list])


if __name__ == '__main__':
    import torch_geometric

    dataset = 'Cora'
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', dataset)
    dataset = Planetoid(path, dataset, T.NormalizeFeatures())
    data = dataset[0]


    class Net(torch.nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = GATConv(dataset.num_features, 8, heads=8, dropout=0.6)
            # On the Pubmed dataset, use heads=8 in conv2.
            self.conv2 = GATConv(
                8 * 8, dataset.num_classes, heads=1, concat=True, dropout=0.6)

        def forward(self):
            x = F.dropout(data.x, p=0.6, training=self.training)
            x = F.elu(self.conv1(x, data.edge_index))
            x = F.dropout(x, p=0.6, training=self.training)
            x = self.conv2(x, data.edge_index)
            return F.log_softmax(x, dim=1)


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, data = Net().to(device), data.to(device)
    gnn_model_summary(model)
