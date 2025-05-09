import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

cmaps = 'utils/cmaps'

def load_custom_cmaps():
    # Fluent rainbow
    colors = np.loadtxt(f'{cmaps}/rainbow2_cmap.txt')
    cmap = LinearSegmentedColormap.from_list('rainbow2', np.flipud(colors), N=len(colors))
    mpl.colormaps.register(cmap, name='rainbow2')
