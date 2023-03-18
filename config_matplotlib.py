# https://stackoverflow.com/a/39566040
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np

SMALL_SIZE = 12
MEDIUM_SIZE = 14
BIGGER_SIZE = 16

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


def get_sequential_colormap(color):
    N = 256
    vals = np.ones((N, 4))
    vals[:, 0] = np.linspace(1, color[0], N)
    vals[:, 1] = np.linspace(1, color[1], N)
    vals[:, 2] = np.linspace(1, color[2], N)
    return ListedColormap(vals)