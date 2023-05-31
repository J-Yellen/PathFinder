#!/usr/bin/env python3
"""
#####################################
# Part of the PATH FINDER  Module   #
# Author J.Yellen                   #
#####################################
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import ListedColormap
from matplotlib import cm
from .matrix_handler import BinaryAcceptance
from .result import Results

plt.rcParams['legend.handlelength'] = 1
plt.rcParams['legend.handleheight'] = 1

plt.rcParams['savefig.dpi'] = 300
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] =  ["Palatino"]
plt.rcParams["mathtext.fontset"] = "cm"
#plt.rcParams["axes.grid"] = False

def set_legend(ax:plt.Axes)->None:
    legend_elements = [Rectangle((0.0, 0.0), 1, 2, facecolor='w', edgecolor='k',
                            label=r'$\rho_{ij} < T$'),
                    Rectangle((0.0, 0.0), 1, 2, facecolor='k', edgecolor='k',
                            label=r'$\rho_{ij} \geq T$'),
                    Rectangle((0.0, 0.0), 1, 2, facecolor='darkgrey', edgecolor='k',
                            label=r'$\rho_{ii}$')
                          ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize='30', framealpha=1)

def add_cell_borders(ax:plt.Axes, dim:int)->None:
    for i in range(0, dim):
        ax.axhline(y=i+0.5, xmin=0, xmax=(i+2)/dim, linewidth=2, color="k")
        ax.axvline(x=i+0.5, ymin=0, ymax=(dim-i)/dim, linewidth=2, color="k")

def format_path(path:list)->np.ndarray:
    #[0, 1, 2, 3]
    #x = [[0, 1], [1, 1], [1, 2], [2, 2], [2, 3], [3, 3]]
    #y = [[0, 0], [0, 1], [1, 1], [1, 2], [2, 2], [2, 3]]
    xval, yval = [], []
    for i, j in zip(path[:-1:], path[1::]):
        xval += [[i, i], [i, j]]
        yval += [[i, j], [j, j]]
    return [xval, yval]

def make_path0(paths:list, shift:bool=True)->dict:

    shift_i = 1./(len(paths)+1)
    lshft = 0.5
    c_vals = np.linspace(0.0, 1, len(paths))
    color = cm.rainbow(c_vals)
    ret = {}
    for i, p in enumerate(paths):
        ret[i] = {}
        posx, posy, colors = [], [], []
        if shift:
            #lshft = track[p[0]]*shift_i
            lshft = (i+1)*shift_i
        pth = format_path(p)
        for j, k in zip(pth[0], pth[1]):
            xy = np.array([j, k])
            loc = abs(c_vals - (1-i/len(paths))).argmin()
            posx += [list(xy[0]+lshft-0.5)]
            posy += [list(xy[1]+lshft-0.5)]
            colors += [color[loc]]
        ret[i]['x'] = posx
        ret[i]['y'] = posy
        ret[i]['color'] = colors
    return ret

def add_results(ax:plt.Axes, res:Results, lim:int)->None:
    for i, item in make_path0(res.get_paths[:lim:], shift=True).items():
        for x, y, c in zip(item['x'], item['y'], item['color']):
            ax.plot(x, y, lw=3, color=c, linestyle='-')

def plot(bam:BinaryAcceptance, result:Results|None=None, top:int=None)-> plt.Axes:

    cmap = ListedColormap(['k', 'darkgrey', 'lightgrey', 'w'], name='bwg')
    dat = np.array(bam.bin_acc, dtype=float, copy=True)
    dat[np.diag_indices(dat.shape[0])] = 0.3
    dat[np.triu_indices(dat.shape[0], k=1)] = 0.6
    #_, axis = plt.subplots(figsize=(24,16))
    _, axis = plt.subplots(figsize=(16, 16/1.5))
    axis.imshow(dat, cmap=cmap)
    axis.set_xticks([])
    axis.set_yticks([])
    add_cell_borders(axis, bam.dim)
    set_legend(axis)
    if result:
        if top is None: top = result._top
        add_results(axis, result, lim=top)
    return axis
