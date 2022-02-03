
__author__ = 'SherlockLiao'

import os

import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import save_image
from time import time

from signor.ioio.dir import make_dir
from signor.viz.pt.layers import save_act_across_layers
from signor.utils.pt.shape import dataloader_shape, ReshapeTransform
from signor.monitor.probe import summary
from signor.monitor.time import timefunc


if not os.path.exists('./mlp_img'):
    os.mkdir('./mlp_img')


def to_img(x):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 1, 28, 28)
    return x


num_epochs = 100
batch_size = 128
learning_rate = 1e-3
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dev = 'cpu'

# img_transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
# ])

img_transform = transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,)),
                           ReshapeTransform((-1, 28*28))
                       ])

dataset = MNIST('./data_', transform=img_transform, download=True)

dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

print(dataset)
print(dataloader)
dataloader_shape(dataloader)
exit()

# use for saving itermediate results, of shape [128, 1, 784].
# https://github.com/pytorch/pytorch/issues/1512#issuecomment-405015099
# a more complicated way. didn't try here.
_dataloader = DataLoader(dataset, batch_size=256, shuffle=True, )

class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.ReLU(True),
            nn.Linear(128, 64),
            nn.ReLU(True), nn.Linear(64, 12), nn.ReLU(True), nn.Linear(12, 3))


        self.decoder = nn.Sequential(
            nn.Linear(3, 12),
            nn.ReLU(True),
            nn.Linear(12, 64),
            nn.ReLU(True),
            nn.Linear(64, 128),
            nn.ReLU(True), nn.Linear(128, 28 * 28), nn.Tanh())

    @timefunc
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


model = autoencoder().to(dev)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(
    model.parameters(), lr=learning_rate, weight_decay=1e-5)

for data in dataset:
    img, _ = data
    img = img.view(img.size(0), -1)

save_kwargs = dict()
save_kwargs['dir'] =  None # '/Users/admin/Documents/osu/Research/Signor/data_/autoencoder/'
save_kwargs['epoch'] = 1
make_dir(save_kwargs['dir'])

# save_act_across_layers(net = model, input = pt.rand((300, 1, 28*28)).to(dev), save=True, **save_kwargs)

# exit()

# train
for epoch in range(num_epochs):
    t_start = time()
    total_loss = 0
    save_kwargs['epoch'] = epoch
    for i, data in enumerate(dataloader):
        save_kwargs['batch'] = i
        img, _ = data
        # img = img.view(img.size(0), -1, 28*28)
        img = Variable(img).to(dev)
        # ===================forward=====================
        output = model(img)
        loss = criterion(output, img)
        # ===================backward====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    t_end = time()
    save_act_across_layers(net=model, dataloader=_dataloader, dev = dev, save=True, **save_kwargs)

    # from signor.ml.pt.graph.parameter import compose_param_groups
    # compose_param_groups(model)
    # exit()

    # ===================log========================
    print('epoch [{}/{}], loss:{:.4f}. \t time: {:.1f}'.format(epoch + 1, num_epochs, total_loss, t_end - t_start))
    if epoch % 10 == 0:
        pic = to_img(output.cpu().data)
        save_image(pic, './mlp_img/image_{}.png'.format(epoch))
    del img, loss

torch.save(model.state_dict(), './sim_autoencoder.pth')