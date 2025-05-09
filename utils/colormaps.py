import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

cmaps = 'utils/cmaps'

def load_custom_cmaps():
    # Fluent rainbow
    colors = np.loadtxt(f'{cmaps}/fluent_rainbow_cmap.txt')
    cmap = LinearSegmentedColormap.from_list('fluent_rainbow', np.flipud(colors), N=len(colors))
    mpl.colormaps.register(cmap, name='fluent_rainbow')
