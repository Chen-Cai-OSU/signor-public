# Created at 2020-07-20
# Summary: visualization tool for error analysis for ml regression

import matplotlib.pyplot as plt
import numpy as np
import torch
from signor.monitor.probe import summary

plt.locator_params(nbins=4)

def tonp(tsr):
    if isinstance(tsr, np.ndarray):
        return tsr

    assert isinstance(tsr, torch.Tensor)
    tsr = tsr.cpu()
    assert isinstance(tsr, torch.Tensor)
    try:
        arr = tsr.numpy()
    except:
        arr = tsr.detach().numpy()
    assert isinstance(arr, np.ndarray)
    return arr

def viz_reg_error(y_pred, y, title=None, show=False, save=None, save_data=None):
    y_pred, y = tonp(y_pred), tonp(y)

    f, ax = plt.subplots()
    ax.scatter(y_pred, y, c=".3")
    add_identity(ax, color='r', ls='--')

    ax.set_xlabel('Calculated Value')
    ax.set_ylabel('Predicted Value')

    if save_data:
        data = np.vstack((y, y_pred)).T
        summary(data, 'data', highlight=True)
        np.save(save_data, data)
        print(f'save data at {save_data}')
    if title: ax.set_title(title)
    if save: plt.savefig(save, bbox_inches='tight')
    if show: plt.show()


def add_identity(axes, *line_args, **line_kwargs):
    # https://bit.ly/3g3Zayd
    identity, = axes.plot([], [], *line_args, **line_kwargs)
    def callback(axes):
        low_x, high_x = axes.get_xlim()
        low_y, high_y = axes.get_ylim()
        low = max(low_x, low_y)
        high = min(high_x, high_y)
        identity.set_data([low, high], [low, high])
    callback(axes)
    axes.callbacks.connect('xlim_changed', callback)
    axes.callbacks.connect('ylim_changed', callback)
    return axes

if __name__ == '__main__':

    y_pred = 10 * np.random.random((100,))
    y = y_pred + 0.01 * np.random.random((100,))
    viz_reg_error(y_pred, y, show=True)