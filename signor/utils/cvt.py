""" convert between different formats """
import os
from signor.monitor.probe import summary

from signor.ioio.dir import assert_file_exist
from signor.monitor.time import tf

from PIL import Image
from torchvision import transforms


def jpnb2script():
    """
    jupyter nbconvert --to script [YOUR_NOTEBOOK].ipynb
    convert jupyter notebook to python script
    :return:
    """

    jupyter = '~/anaconda3/bin/jupyter nbconvert'
    args = '--to script'
    dir = '/Users/admin/Documents/osu/Research/Signor/signor/viz/pt/'
    nb = 'pytorchviz.ipynb'

    cmd = ' '.join([jupyter, args, dir + nb])
    print(cmd)
    os.system(cmd)

@tf
def png2tsr(img_link, size=480):
    """

    :param img_link:
    :param size: int or (x, y)
    :return:
    """
    assert_file_exist(img_link)
    img = Image.open(img_link)
    trans = transforms.Compose([
        transforms.CenterCrop(size),
        transforms.ToTensor(),
    ])

    if isinstance(size, int):
        t = trans(img).reshape((4, size, size))
        t = t[:3, :, :]
        assert t.size() == (3, size, size)
        return t
    else:
        NotImplementedError


if __name__ == '__main__':
    lk = '/home/cai.507/Dropbox/2020_Spring/Network/proj/data/TianXie/cif/mp-ids-3402/band_gap/mp-1482.png'
    summary(png2tsr(lk, size=400))

    exit()
    jpnb2script()
