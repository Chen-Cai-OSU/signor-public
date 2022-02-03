""" from tensor to img. used for debug """
import glob

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
from PIL import Image
from signor.monitor.probe import summary
import torchvision


def tsr2img(tsr):
    if isinstance(tsr, torch.Tensor):
        tsr = tsr.numpy()
    else:
        assert isinstance(tsr, np.ndarray)
    imgplot = plt.imshow(tsr)
    plt.show()


# Plot the image here using matplotlib.
def plot_image(tensor):
    """ works for 3-dim tensor """
    plt.figure()
    # imshow needs a numpy array with the channel dimension
    # as the the last dimension so we have to transpose things.
    plt.imshow(tensor.numpy().transpose(1, 2, 0))
    plt.show()

def img_batch(reshape=False):
    """ generate a small batch of images for testing activations across layers
     return tensor of shape (500, 3, 224, 224)
     """

    # todo: fix randomness
    img_dir = '/home/cai.507/Documents/DeepLearning/Signor/data/tiny-imagenet-200/train/n01443537/images/'
    img_id = 'n01443537_140.JPEG'

    pil2tensor = transforms.Compose([transforms.RandomResizedCrop(224), transforms.ToTensor()])
    images = glob.glob(f"{img_dir}*JPEG")  # a list of len 500

    imgs = []
    for img in images:
        pil_image = Image.open(img)
        rgb_image = pil2tensor(pil_image)
        if reshape:
            rgb_image = torch.unsqueeze(rgb_image, 0)
        imgs.append(rgb_image)

    imgs = torch.cat(imgs, dim=0)
    return imgs

def single_image(reshape=False):
    img_dir = '/home/cai.507/Documents/DeepLearning/Signor/data/'

    pil2tensor = transforms.Compose([transforms.RandomResizedCrop(224), transforms.ToTensor()])
    images = glob.glob(f"{img_dir}*jpg")  # a list of len 500
    print(images)
    imgs = []
    for img in images:
        pil_image = Image.open(img)
        rgb_image = pil2tensor(pil_image)
        if reshape:
            rgb_image = torch.unsqueeze(rgb_image, 0) # return torch.Size([1, 3, 224, 224])
        imgs.append(rgb_image)

    imgs = torch.cat(imgs, dim=0)
    return imgs


if __name__ == '__main__':
    imgs = img_batch(reshape=True)

    grid_img = torchvision.utils.make_grid(imgs, nrow=25)
    plt.imshow(grid_img.permute(1, 2, 0))
    plt.show()

    exit()

    exit()

    tsr = torch.rand((250, 250))
    tsr2img(tsr)

    exit()

    exit()
