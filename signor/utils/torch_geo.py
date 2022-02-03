from signor.utils.random_ import fix_seed

fix_seed()
import networkx as nx
import numpy as np
import torch
import torch_geometric

from signor.utils.np import tonp

torch.set_default_dtype(torch.float64)

from torch_geometric.data import Data
from torch_geometric.utils import to_networkx

from signor.monitor.probe import summary


def examine_list_of_data(DataList, n_sample=1, attribute=None):
    """ examine the list of torch_geometric.data.data.Data by a attribute"""

    DataList = [DataList[i] for i in range(len(DataList))]  # dataset xyz (100)
    assert isinstance(DataList, list)
    for x in DataList:
        assert isinstance(x, torch_geometric.data.data.Data)

    # subsample DataList
    n_sample = min(n_sample, len(DataList))
    random_indices = np.random.randint(0, len(DataList), size=n_sample).tolist()
    sub_DataList = [DataList[i] for i in random_indices]

    # attribute
    feats = []
    for data in sub_DataList:
        try:
            tmp = data.__getattribute__(attribute)  # an tensor/array
        except AttributeError:
            raise Exception(f'no attribute {attribute} in {data}')
        feats.append(tonp(tmp))
    feats = np.concatenate(feats, axis=0)

    print(f'Concatenated attributes {attribute} of {n_sample} torch_geometric.data.data.Data')
    summary(feats)


def random_edge_index(n_edge=200, n_node=20):
    """ generate random edge tensor of shape (2, n_edge) """
    assert n_edge % 2 == 0
    assert n_edge <= n_node * (n_node - 1), f'n_edge: {n_edge}; n_node: {n_node}'
    edges = []
    for i in range(n_edge // 2):
        a, b = np.random.choice(n_node, 2, replace=False).tolist()
        while (a, b) in edges:
            a, b = np.random.choice(n_node, 2, replace=False).tolist()
        edges.append((a, b))
        edges.append((b, a))
    edges = list(edges)
    edges = torch.LongTensor(np.array(edges).T)
    return edges


def random_pygeo_graph(n_node, node_feat_dim, n_edge, edge_feat_dim, device='cpu', viz=False):
    """ random DIRECTED pyG graph """
    g = Data(x=torch.rand(n_node, node_feat_dim),
             edge_index=random_edge_index(n_edge, n_node),
             edge_attr=(1000*torch.rand(n_edge, edge_feat_dim)).type(torch.LongTensor),
             edge_weight=torch.ones(n_edge))

    g_nx = to_networkx(g).to_undirected()
    n_compoent = nx.number_connected_components(g_nx)
    assert n_compoent == 1, f'number of component is {n_compoent}'
    g = g.to(device)
    return g


if __name__ == '__main__':
    # data = Data(x=torch.rand(100, 3), cif=torch.Tensor(['mp-1']*100))
    data = Data(x=torch.rand(100, 3))
    DataList = [data] * 5
    examine_list_of_data(DataList, n_sample=3, attribute='x')
