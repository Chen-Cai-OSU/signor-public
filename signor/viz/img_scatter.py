# https://stackoverflow.com/questions/22566284/matplotlib-how-to-plot-images-instead-of-points
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.cbook import get_sample_data
from signor.ioio.dir import find_files

np.random.seed(42)

def getImage(path, zoom=.1):
    return OffsetImage(plt.imread(path), zoom=zoom)


def scatter_with_diff_imgs(zoom=0.2):
    read_dir = '/Users/admin/Documents/osu/Research/Signor/signor/viz/cif_test/'
    files = find_files(read_dir, suffix='.png', include_dir=True)
    paths = files[:1]
    print(paths)

    x = np.random.random(1)
    y = np.random.random(1)

    fig, ax = plt.subplots()
    ax.scatter(x, y)

    for x0, y0, path in zip(x, y, paths):
        ab = AnnotationBbox(getImage(path, zoom=zoom), (x0, y0), frameon=False)
        ax.add_artist(ab)
    # plt.show()
    plt.savefig('/Users/admin/Documents/osu/Research/Signor/data/material/scatter.png')

def main():
    # same image at different locations
    x = np.linspace(0, 10, 20)
    y = np.cos(x)
    image_path = get_sample_data('ada.png')
    print('image_path', type(image_path))


    fig, ax = plt.subplots()
    imscatter(x, y, image_path, zoom=0.03, ax=ax)
    ax.plot(x, y)
    plt.show()

def imscatter(x, y, image, ax=None, zoom=1):
    if ax is None:
        ax = plt.gca()
    image = plt.imread(image) # an array
    print('plt.imread', type(image))

    im = OffsetImage(image, zoom=zoom)
    x, y = np.atleast_1d(x, y)
    artists = []
    for x0, y0 in zip(x, y):
        ab = AnnotationBbox(im, (x0, y0), xycoords='data', frameon=False)
        artists.append(ax.add_artist(ab))
    ax.update_datalim(np.column_stack([x, y]))
    ax.autoscale()
    return artists

if __name__ == '__main__':
    scatter_with_diff_imgs()
    # main()