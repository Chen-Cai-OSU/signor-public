from signor.datasets.image_classification.mnist import load_mnist
from signor.monitor.probe import summary
from signor.viz.dim_red import dim_reducer
import matplotlib.pyplot as plt
from signor.utils.np import add_noise
import sys
import argparse

parser = argparse.ArgumentParser(description='Noise Stabability')
parser.add_argument('--noise_dim', default=0, type=int, help='noise dim')
parser.add_argument('--method', default='tsne', type=str, help='dim reducer method', choices=['umap', 'tsne'])

if __name__ == '__main__':
    args = parser.parse_args()
    data_dir = '/home/cai.507/Documents/DeepLearning/Signor/data/'
    _, _, data = load_mnist(data_dir)
    summary(data, 'data')

    x, y = data
    x = add_noise(x, args.noise_dim)
    dr = dim_reducer()
    x, y = dr.sampler((x, y), s=500)
    if args.method == 'tsne':
        pos = dr.tsne(x)
    else:
        pos = dr.umap(x)

    dr.umaps(x, y)
    sys.exit()
    cmap, norm, bounds = dr.set_color(y, discrete=True)
    plt.scatter(pos[:, 0], pos[:, 1], c=y, cmap=cmap, norm=norm)
    plt.title(f'{args.method}: noise dim {args.noise_dim}')
    plt.show()
