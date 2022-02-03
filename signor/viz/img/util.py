import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

from signor.monitor.probe import summary
from signor.viz.img.imshow import batch_show

if __name__ == '__main__':

    # https://stackoverflow.com/questions/35355930/matplotlib-figure-to-image-as-a-numpy-array
    fig = Figure()
    canvas = FigureCanvas(fig)
    ax = fig.gca()

    pos = np.random.rand(100, 2)
    ax.scatter(pos[:, 0], pos[:, 1])
    ax.axis('off')

    canvas.draw()  # draw the canvas, cache the renderer

    width, height = fig.get_size_inches() * fig.get_dpi()
    width, height = int(width), int(height)
    print(width, height)
    image = np.fromstring(canvas.tostring_rgb(), dtype='uint8').reshape(height, width, 3)

    summary(image, 'image')
    images = [image] * 16

    batch_show(images)

    # arr = 255 * np.random.random((16, 200, 200, 3))
    # arr = arr.astype('uint8')
    # batch_show(arr)



