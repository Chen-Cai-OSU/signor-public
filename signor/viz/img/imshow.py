#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : imshow.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 01/19/2018
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.
from PIL import Image
from os import listdir, path
from signor.ioio.dir import find_files

import numpy as np

from signor.viz.img.imgio import imshow as imshow_
from signor.viz.img.imgproc import resize_minmax
from signor.viz.img.imgrid import image_grid


def imshow(img, resize=(600, 800), title='imshow'):
    """
    Image show with different parameter order.

    :param img: Image.
    :param resize: Resize factor, a tuple (min_dim, max_dim).
    :param title: The title of the shown window.
    """
    img = resize_minmax(img, *resize, interpolation='NEAREST')
    imshow_(title, img)


def batch_show(batch, nr_show=16, grid_desc=('4v', '4h'), resize=(600, 800), title='batch_show'):
    """
    Show a batch of images.

    :param batch: The batched data_: can be either a ndarray of shape (batch_size, h, w, c) or a list of images.
    :param nr_show: Number of images to be displayed. Default set to be 16.
    :param grid_desc: Grid description. See `tartist.image.image_grid` for details.
    :param resize: Resize factor, a tuple (min_dim, max_dim).
    :param title: The title of the shown window.
    """

    batch = batch[:nr_show]
    batch = np.array(batch)

    if len(batch) < 16:
        batch = np.concatenate([
            batch,
            np.zeros([16 - len(batch), batch.shape[1], batch.shape[2], batch.shape[3]], dtype=batch.dtype)
        ], axis=0)

    img = image_grid(batch, grid_desc)
    img = resize_minmax(img, *resize, interpolation='NEAREST')
    imshow_(title, img)


from PIL import Image, ImageChops

def trim(f):
    """
    trim image
    :param f: img path: "/Users/admin/Documents/osu/Research/Signor/signor/viz/img/test.jpg"
    :return: save trimed one in orginal f
    """
    im = Image.open(f)
    bg = Image.new(im.mode, im.size, im.getpixel((0,0)))
    diff = ImageChops.difference(im, bg)
    diff = ImageChops.add(diff, diff, 2.0, -100)
    bbox = diff.getbbox()
    if bbox:
        return im.crop(bbox)
    im.save(f)

def stitch_imgs(imgs_list, out_f ='test.png'):
    """
    https://stackoverflow.com/questions/30227466/combine-several-images-horizontally-with-python
    :param imgs_list: a list of image file (['Test1.jpg', 'Test2.jpg', 'Test3.jpg'])
    :return:
    """

    n_img = len(imgs_list)
    n_row, n_col = int(np.ceil(np.sqrt(n_img))), int(np.ceil(np.sqrt(n_img)))
    # [trim(f) for f in imgs_list]

    images = [Image.open(x) for x in imgs_list]
    widths, heights = zip(*(i.size for i in images))
    total_width = max(widths) * n_col
    total_height = max(heights) * n_row

    new_im = Image.new('RGB', (total_width, total_height))

    x_offset = 0
    for i in range(n_row):
        y_offset = 0
        for j in range(n_col):
            idx = i * n_col + j
            if idx >= n_img: break
            im = images[idx]
            new_im.paste(im, (x_offset, y_offset))
            y_offset += im.size[1]
        x_offset += im.size[0]

    new_im.save(out_f)


if __name__ == '__main__':
    # read_dir = '/Users/admin/Documents/osu/Research/Signor/data/material/'
    # read_dir = '/home/cai.507/Documents/DeepLearning/Signor/data/material/'
    # read_dir = '/home/cai.507/Documents/DeepLearning/Signor/signor/viz/cif_test/'
    read_dir = '/Users/admin/Documents/osu/Research/Signor/signor/viz/cif_test/'
    files = find_files(read_dir, suffix='.png', include_dir=True)
    im_path_list = files[:100]
    stitch_imgs(im_path_list, out_f='test.png')
    exit()

    f = "/Users/admin/Documents/osu/Research/Signor/signor/viz/img/test.jpg"
    f_trim = "/Users/admin/Documents/osu/Research/Signor/signor/viz/img/test_trim.jpg"
    im = Image.open(f)
    im = trim(im)
    im.save(f_trim)

    exit()

    arr = 255*np.random.random((16, 200, 200, 2))
    arr = arr.astype('uint8')
    batch_show(arr)

