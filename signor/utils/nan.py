nan1 = 0.12345
nan2 = -0.12345


def assert_nonan(x):
    import torch

    res = torch.isnan(x)
    assert (res == False).all(), 'contains Nan'

if __name__ == '__main__':
    import torch
    x = torch.tensor([1, float('nan'), 2])
    assert_nonan(x)

