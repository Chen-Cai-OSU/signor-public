# Created at 2020-07-20
# Summary: tex format

from matplotlib import rc

# https://bit.ly/3cv2mkB
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
# for Palatino and other serif fonts use:
rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)
