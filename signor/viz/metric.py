# Created at 4/27/21
# Summary: viz train/test loss

import textwrap

from signor.format.format import red


def plot_train_test_loss(train, test=None, title=None, path=False, viz=False, scale='linear'):
    # plt.plot(train, label='train')
    plot_smooth_curve(train, label='train')
    if test is not None:
        plot_smooth_curve(test, label='test')
        # plt.plot(test, label='test')

    if title is not None:
        title = "\n".join(textwrap.wrap(str(title)))
        plt.title(title, fontsize=14)
    else:
        plt.title('train/test loss')
    plt.legend()
    plt.yscale(scale)

    if path:
        print(red(f'save figure at {path}'))
        plt.savefig(path, bbox_inches='tight')

    if viz: plt.show()
    return


import numpy             as np
import pandas            as pd
import matplotlib.pyplot as plt


def plot_smooth_curve(data, show=False, n_steps=5, label=''):
    time_series_array = data # np.sin(np.linspace(-np.pi, np.pi, 400)) + np.random.rand((400))
    # n_steps = 5  # number of rolling steps for the mean/std.

    time_series_df = pd.DataFrame(time_series_array)
    smooth_path = time_series_df.rolling(n_steps).mean()
    path_deviation = 2 * time_series_df.rolling(n_steps).std()

    under_line = (smooth_path - path_deviation)[0]
    over_line = (smooth_path + path_deviation)[0]

    # Plotting:
    plt.plot(smooth_path, linewidth=2, label=label)  # mean curve.
    plt.fill_between(path_deviation.index, under_line, over_line,  alpha=.4)  # std curves.
    if show: plt.show()

def generate_noise(data, n):
    """ generate fake noise """
    noise_data = np.array([np.random.random() for _ in range(100)]) * np.std(data) * 0.5
    return noise_data



if __name__ == '__main__':
    n_epoch = 100
    # train, test = [torch.rand(1) for _ in range(n_epoch)], [torch.rand(1) for _ in range(n_epoch)]

    train = 2 * np.array([np.exp(0.31 * -i) for i in range(n_epoch)])
    test = np.array([1.2 * np.exp(-0.55 * i) + 0.23 for i in range(n_epoch)])

    noise_train = generate_noise(train, n_epoch)
    noise_test = generate_noise(test, n_epoch)
    plot_train_test_loss(train + noise_train, test=test + noise_test, title='train-test', viz=True)
