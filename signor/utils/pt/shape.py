from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from signor.monitor.probe import summary
import torch
from torch.utils import data
from signor.utils.np import tonp


def dataloader_shape(dl):
    assert isinstance(dl, torch.utils.data.DataLoader), 'Input has to be a DataLoader'
    for batch in dl:
        break
    summary(batch, 'batch')


class ReshapeTransform:
    def __init__(self, new_size):
        self.new_size = new_size

    def __call__(self, img):
        return torch.reshape(img, self.new_size)


# todo: build a dataloader for any shape. can be useful for testing
class fake_Dataset(data.Dataset):
    'Characterizes a dataset for PyTorch'

    def __init__(self, x, y):
        'Initialization'
        self.x = x
        self.y = y

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.x)

    def __getitem__(self, index):
        X = self.x[index]  # pt.load('data_/' + ID + '.pt')
        y = self.y[index]

        return X, y


def lable_cnt(x):
    x = tonp(x)
    assert x.ndim == 1
    cnt = len(set(x))
    print(f'Label count for tesnor of shape {x.shape} is {cnt}')


if __name__ == '__main__':
    x = torch.randint(0, 100, (1000,))
    lable_cnt(x)
    exit()

    x = [torch.rand(5, 3)] * 100
    y = [1] * 100
    dl = torch.utils.data.DataLoader(fake_Dataset(x, y), batch_size=10)
    dataloader_shape(dl)
    exit()

    x = torch.cat(x, dim=0)
    summary(x, 'after cat')

    exit()
    mnist_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        ReshapeTransform((-1, 28 * 28))
    ])

    dataset = datasets.MNIST('./data_', download=True, transform=mnist_transforms)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    dataloader_shape(dataloader)

    from signor.utils.pt.img import tsr2img

    for x, target in dataloader:
        img = x[1].reshape((28, 28))
        tsr2img(img)
    exit()

    dataset = torch.rand(32, 3, 224, 224)
    dataloader = DataLoader(dataset, batch_size=32)
    dataloader_shape(dataloader)
