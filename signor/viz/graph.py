# matplotlib.use('tkagg')
import collections
import os
import random
import sys

import matplotlib.pyplot as plt

# from importlib import reload  # Python 3.4+ only.
# import matplotlib
# from mpl_toolkits.mplot3d import Axes3D
from signor.ioio.dir import sig_dir, make_dir
from signor.monitor.time import tf


def viz_graph(g, node_size=5, edge_width=1, node_color='b', color_bar=False, show=False):
    # g = nx.random_geometric_graph(100, 0.125)
    pos = nx.spring_layout(g)
    nx.draw(g, pos, node_color=node_color, node_size=node_size, with_labels=False, width=edge_width)
    if color_bar:
        # https://stackoverflow.com/questions/26739248/how-to-add-a-simple-colorbar-to-a-network-graph-plot-in-python
        sm = plt.cm.ScalarMappable(norm=plt.Normalize(vmin=min(node_color), vmax=max(node_color)))
        sm._A = []
        plt.colorbar(sm)
    if show: plt.show()


def test():
    G = nx.star_graph(20)
    pos = nx.spring_layout(G)
    colors = range(20)
    cmap = plt.cm.Blues
    vmin = min(colors)
    vmax = max(colors)
    nx.draw(G, pos, node_color='#A0CBE2', edge_color=colors, width=4, edge_cmap=cmap,
            with_labels=False, vmin=vmin, vmax=vmax)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm._A = []
    plt.colorbar(sm)
    plt.show()


def sample():
    G = nx.random_geometric_graph(200, 0.125)
    pos = nx.get_node_attributes(G, 'pos')

    # find node near center (0.5,0.5)
    dmin = 1
    ncenter = 0
    for n in pos:
        x, y = pos[n]
        d = (x - 0.5) ** 2 + (y - 0.5) ** 2
        if d < dmin:
            ncenter = n
            dmin = d

    # color by path length from node near center
    p = dict(nx.single_source_shortest_path_length(G, ncenter))

    plt.figure(figsize=(8, 8))
    nx.draw_networkx_edges(G, pos, nodelist=[ncenter], alpha=0.4)
    nx.draw_networkx_nodes(G, pos, nodelist=list(p.keys()),
                           node_size=8,
                           node_color=list(p.values()),
                           cmap=plt.cm.Reds_r)

    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.05, 1.05)
    plt.axis('off')
    plt.show()


def viz_deghis(G):
    # G = nx.gnp_random_graph(100, 0.02)
    degree_sequence = sorted([d for n, d in G.degree()], reverse=True)  # degree sequence
    # print "Degree sequence", degree_sequence
    degreeCount = collections.Counter(degree_sequence)
    deg, cnt = zip(*degreeCount.items())

    fig, ax = plt.subplots()
    plt.bar(deg, cnt, width=0.80, color='b')

    plt.title("Degree Histogram")
    plt.ylabel("Count")
    plt.xlabel("Degree")
    ax.set_xticks([d + 0.4 for d in deg])
    ax.set_xticklabels(deg)

    plt.show()


def viz_mesh_graph(edge_index, pos, viz_flag=True):
    """
    :param edge_index: np.array of shape (2, 2*num_edge)
    :param pos: np.array of shape (n_pts, 3)
    :return: viz mesh graph
    """
    n_node = pos.shape[0]
    n_edge = edge_index.shape[1] // 2

    edges = edge_index  # np.array([[ 0,  0,  1,  1 ], [ 1,  9,  0,  2]])
    edges_lis = list(edges.T)
    edges_lis = [(edge[0], edge[1]) for edge in edges_lis]

    pos_dict = dict()
    for i in range(n_node):
        pos_dict[i] = tuple(pos[i, :])

    g = nx.from_edgelist(edges_lis)
    for node, value in pos_dict.items():
        g.node[node]['pos'] = value
    assert len(g) == n_node
    assert len(g.edges()) == n_edge
    if not viz_flag: return g

    nx.draw(g, pos_dict)
    plt.show()


def generate_random_3Dgraph(n_nodes, radius, seed=None):
    if seed is not None:
        random.seed(seed)

    # Generate a dict of positions
    pos = {i: (random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1)) for i in range(n_nodes)}

    # Create random 3D network
    G = nx.random_geometric_graph(n_nodes, radius, pos=pos)

    return G


import networkx as nx
import numpy as np


def viz_3dnx():
    from mayavi import mlab
    # https://bit.ly/3aNb7YO

    # some graphs to try
    # H=nx.krackhardt_kite_graph()
    # H=nx.Graph();H.add_edge('a','b');H.add_edge('a','c');H.add_edge('a','d')
    # H=nx.grid_2d_graph(4,5)
    H = nx.cycle_graph(20)

    # reorder nodes from 0,len(G)-1
    G = nx.convert_node_labels_to_integers(H)
    # 3d spring layout
    pos = nx.spring_layout(G, dim=3)
    # numpy array of x,y,z positions in sorted node order
    xyz = np.array([pos[v] for v in sorted(G)])
    # scalar colors
    scalars = np.array(list(G.nodes())) + 5

    pts = mlab.points3d(
        xyz[:, 0],
        xyz[:, 1],
        xyz[:, 2],
        scalars,
        scale_factor=0.1,
        scale_mode="none",
        colormap="Blues",
        resolution=20,
    )

    pts.mlab_source.dataset.lines = np.array(list(G.edges()))
    tube = mlab.pipeline.tube(pts, tube_radius=0.01)
    mlab.pipeline.surface(tube, color=(0.8, 0.8, 0.8))
    mlab.show()


@tf
def plot3d_graph(ax, G, edgec='black', edgealpha=0.25):
    """ plot 3d graph """

    pos = nx.get_node_attributes(G, 'pos')
    n = G.number_of_nodes()
    edge_max = max([G.degree(i) for i in range(n)])

    # Define color range proportional to number of edges adjacent to a single node
    # colors = [plt.cm.plasma(G.degree(i) / edge_max) for i in range(n)]

    with plt.style.context(('ggplot')):
        # Loop on the pos dictionary to extract the x,y,z coordinates of each node
        for key, value in pos.items():
            xi, yi, zi = value
            if key == 1:
                c, size = 'red', 130
            elif key == 20:
                c, size = 'orange', 130
            else:
                c, size = 'black', 1
            ax.scatter(xi, yi, zi, s=size, edgecolors='k', alpha=0.7, c=c)  # c=colors[key],

        # Loop on the list of edges to get the x,y,z, coordinates of the connected nodes
        # Those two points are the extrema of the line to be plotted
        for i, j in G.edges():
            pi, pj = pos[i], pos[j]
            x = np.array((pi[0], pj[0]))
            y = np.array((pi[1], pj[1]))
            z = np.array((pi[2], pj[2]))

            # Plot the connecting lines
            ax.plot(x, y, z, c=edgec, alpha=edgealpha)


def cmp_3d_graphs(G1, G2=None, save=False, show=False, title=None):
    if G2:
        assert len(G1) < len(G2), f'Expect G1({len(G1)}) has fewer nodes than G2({len(G2)})'
    # https://www.idtools.com.au/3d-network-graphs-python-mplot3d-toolkit/
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection='3d')
    # ax.set_axis_off()
    plot3d_graph(ax, G1, edgec='blue', edgealpha=0.4)
    if G2:
        plot3d_graph(ax, G2, edgec='red', edgealpha=0.2)

    if save:
        params = {'fontsize': 30}
        ax.set_xlabel('$X$', **params)
        ax.set_ylabel('$Y$', **params)
        ax.set_zlabel('$Z$', **params)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])

        dir = os.path.join(sig_dir(), 'graph', 'permeability', 'paper', 'viz')
        make_dir(dir)
        f = os.path.join(dir, f'{title}.pdf')
        plt.savefig(f, bbox_inches='tight')

    if show:
        # Set the initial view
        angle = 0
        ax.view_init(30, angle)
        ax.set_axis_off()
        plt.show()
    return


def viz_tree(G, with_labels=False, show=False, **kwargs):
    try:
        import pygraphviz
        from networkx.drawing.nx_agraph import graphviz_layout
    except ImportError:
        try:
            import pydot
            from networkx.drawing.nx_pydot import graphviz_layout
        except ImportError:
            raise ImportError("This example needs Graphviz and either "
                              "PyGraphviz or pydot")

    pos = graphviz_layout(G, prog='twopi', args='')
    plt.figure(figsize=(8, 8))
    nx.draw(G, pos, node_size=20, alpha=0.5, node_color="blue", with_labels=with_labels, **kwargs)

    if with_labels == False:
        labels = kwargs['labels']
        print(pos)
        nx.draw_networkx_labels(G, pos, labels, font_color='r', font_size=8)

    plt.axis('equal')
    if show: plt.show()


if __name__ == '__main__':
    n = 5
    g1 = generate_random_3Dgraph(n, 0.5)
    g2 = generate_random_3Dgraph(n, 0.5)
    cmp_3d_graphs(g1, G2=g2, show=True)
    exit()
    viz_3dnx()
    exit()
    G = nx.balanced_tree(3, 5)
    viz_tree(G)
    exit()

    g = nx.random_geometric_graph(100, 0.1)
    viz_graph(g, show=True)

    exit()
    n = 100
    edge_index = np.array([[0, 0, 1, 1, 2, 2],
                           [1, 2, 0, 2, 0, 1]])
    pos = np.array([[1, 2, 3],
                    [2.1, 3, 5],
                    [1, 1, 2.2]])
    # pos = pos[:, 0:2]

    g = viz_mesh_graph(edge_index, pos, viz_flag=False)  # generate_random_3Dgraph(n_nodes=n, radius=0.25, seed=1)
    cmp_3d_graphs(g, 0, save=False)

    sys.exit()
    viz_mesh_graph(edge_index, pos)
    sys.exit()
    test()
