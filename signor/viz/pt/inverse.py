import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from torchvision.models import resnet50

from signor.monitor.probe import summary
from signor.utils.np import one_hot
from signor.format.format import pf
from signor.monitor.stats import stats

inp = torch.rand(1, 3, 224, 224)
inp.requires_grad = True

model = resnet50(pretrained=True)  # nn.Linear(4, 4)
labels = np.array(range(1))
labels = one_hot(labels, 1000)
labels = torch.Tensor(labels)
summary(labels)

# crit = F.nll_loss(output, target) # lambda x: x.sum()
for p in model.parameters(): p.requires_grad = False
optimizer = torch.optim.adam([inp], lr=1)

#TODO: make it work
for iter in range(1, 501):
    stats(inp)
    output = model(inp)

    loss = F.cross_entropy(F.softmax(model(inp)), torch.tensor([1]))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f'Iter: {iter} loss: {pf(loss.item(), 10)}\n')
    input = inp

    if iter %50 ==0:
        grid_img = torchvision.utils.make_grid(input.detach().cpu(), nrow=1)
        plt.imshow(grid_img.permute(1, 2, 0))
        plt.show()
        del input

