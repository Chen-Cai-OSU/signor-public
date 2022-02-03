from signor.monitor.probe import summary

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

class AnimatedScatter(object):
    """An animated scatter plot using matplotlib.animations.FuncAnimation."""
    def __init__(self, numpoints=50):
        self.numpoints = numpoints
        self.stream = self.data_stream()

        # Setup the figure and axes...
        # self.fig, self.ax = plt.subplots()
        self.fig = plt.figure()
        self.ax = Axes3D(self.fig)

        # Then setup FuncAnimation.
        self.ani = animation.FuncAnimation(self.fig, self.update, interval=5, 
                                          init_func=self.setup_plot, blit=True)

    def setup_plot(self):
        """Initial drawing of the scatter plot."""
        x, y, z, s, c = next(self.stream).T
        self.scat = self.ax.scatter(x, y, z, c=c, s=s, vmin=0, vmax=1,
                                    cmap="jet", edgecolor="k")
        self.ax.axis([-10, 10, -10, 10])
        # For FuncAnimation's sake, we need to return the artist we'll be using
        # Note that it expects a sequence of artists, thus the trailing comma.
        return self.scat,

    def data_stream(self):
        """Generate a random walk (brownian motion). Data is scaled to produce
        a soft "flickering" effect."""
        xyz = (np.random.random((self.numpoints, 3))-0.5)*10
        s, c = np.random.random((self.numpoints, 2)).T
        while True:
            xyz += 0.03 * (np.random.random((self.numpoints, 3)) - 0.5)
            s += 0.05 * (np.random.random(self.numpoints) - 0.5)
            c += 0.02 * (np.random.random(self.numpoints) - 0.5)
            # summary(np.c_[xy[:,0], xy[:,1], s, c])
            yield np.c_[xyz[:,0], xyz[:,1], xyz[:,2], s, c]

    def update(self, i):
        """Update the scatter plot."""
        data = next(self.stream)

        # Set x and y data...
        self.scat.set_offsets(data[:, :3])
        # Set sizes...
        # self.scat.set_sizes(300 * abs(data[:, 2])**1.5 + 100)
        # Set colors..
        # self.scat.set_array(data[:, 3])

        # We need to return the updated artist for FuncAnimation to draw..
        # Note that it expects a sequence of artists, thus the trailing comma.
        return self.scat,


if __name__ == '__main__':
    a = AnimatedScatter()
    plt.show()