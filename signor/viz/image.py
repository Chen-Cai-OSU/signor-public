# https://matplotlib.org/tutorials/introductory/images.html
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from os import path

from signor.format.format import filename


def showimg(f):
    assert path.isfile(f), f'file {f} doesn\'t exist'

    f_name = filename(f)
    img = mpimg.imread(f)
    imgplot = plt.imshow(img)
    plt.colorbar()
    plt.title(f_name)
    plt.show()


if __name__ == '__main__':
    f = '/home/cai.507/Documents/DeepLearning/Signor/signor/mlp_img/image_20.png'
    showimg(f)
