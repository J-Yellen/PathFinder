#!/usr/bin/env python3
"""
#####################################
# Part of the PATH FINDER  Module   #
# Author J.Yellen                   #
#####################################
"""
import numpy as np
from copy import copy
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import ListedColormap
from matplotlib import cm
from matplotlib import font_manager
from .matrix_handler import BinaryAcceptance
from .result import Results
from typing import Optional, Tuple

import logging
logging.getLogger('matplotlib.font_manager').disabled = True

plt.rcParams['legend.handlelength'] = 1
plt.rcParams['legend.handleheight'] = 1

plt.rcParams['savefig.dpi'] = 300
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "cm"


fonts = font_manager.findSystemFonts(fontpaths=None, fontext='ttf')
if bool(sum(["Palatino" in f for f in fonts])):
    plt.rcParams["font.serif"] = ["Palatino"]


def set_legend(ax: plt.Axes) -> None:
    legend_elements = [Rectangle((0.0, 0.0), 1, 2, facecolor='w', edgecolor='k', label=r'$\rho_{ij} < T$'),
                       Rectangle((0.0, 0.0), 1, 2, facecolor='k', edgecolor='k', label=r'$\rho_{ij} \geq T$'),
                       Rectangle((0.0, 0.0), 1, 2, facecolor='darkgrey', edgecolor='k', label=r'$\rho_{ii}$')
                       ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize='30', framealpha=1)


def add_cell_borders(ax: plt.Axes, dim: int) -> None:
    for i in range(0, dim):
        ax.axhline(y=i + 0.5, xmin=0, xmax=(i + 2) / dim, linewidth=2, color="k")
        ax.axvline(x=i + 0.5, ymin=0, ymax=(dim - i) / dim, linewidth=2, color="k")


def format_path(path: list) -> np.ndarray:
    # [0, 1, 2, 3]
    # x = [[0, 1], [1, 1], [1, 2], [2, 2], [2, 3], [3, 3]]
    # y = [[0, 0], [0, 1], [1, 1], [1, 2], [2, 2], [2, 3]]
    xval, yval = [], []
    if len(path) > 1:
        for i, j in zip(path[:-1:], path[1::]):
            xval += [[i, i], [i, j]]
            yval += [[i, j], [j, j]]
    else:
        xval = [path]
        yval = [path]
    return [xval, yval]


def make_path0(paths: list, shift: bool = True) -> dict:
    shift_i = 1. / (len(paths) + 1)
    lshft = 0.5
    c_vals = np.linspace(0.0, 1, len(paths))
    color = cm.rainbow(c_vals)
    ret = {}
    for i, p in enumerate(paths):
        ret[i] = {}
        posx, posy = [], []
        if shift:
            lshft = (i + 1) * shift_i
        pth = format_path(p)
        for j, k in zip(pth[0], pth[1]):
            xy = np.array([j, k])
            posx += list(xy[0] + lshft - 0.5)
            posy += list(xy[1] + lshft - 0.5)
        ret[i]['x'] = posx
        ret[i]['y'] = posy
        loc = abs(c_vals - (1 - i / len(paths))).argmin()
        ret[i]['color'] = color[loc]
    return ret


def add_results(ax: plt.Axes, res: Results, lim: int) -> None:
    for i, item in make_path0(res.get_paths[:lim:], shift=True).items():
        if len(item['x']) > 1:
            ax.plot(item['x'], item['y'], lw=3, color=item['color'], linestyle='-',
                    label=f"{res.get_weights[i]:.2g}")
        # else:
        #     ax.scatter(item['x'], item['y'], s=15, color=item['color'], label=f"{res.get_weights[i]:.2g}")


def add_sink_data(bam: BinaryAcceptance, result: Results, xy_labels: Optional[list] = None):
    dat = np.array(bam.bin_acc, dtype=float, copy=True)
    dat = np.insert(dat, bam.dim, True, axis=0)
    dat = np.insert(dat, bam.dim, True, axis=1)
    if xy_labels is None:
        xy_labels = [f'{i}' for i in range(len(dat))] + ['Sink']
    else:
        xy_labels += ['Sink']
    if result is not None:
        sink = [bam.dim]
        result = Results.from_dict({i: {'path': item['path'] + sink,
                                        'weight': item['weight']} for i, item in result.to_dict().items()})
    return dat, result, xy_labels


def plot(bam: BinaryAcceptance, results: Optional[Results] = None, top: Optional[int] = None,
         size: int = 16, xy_labels: Optional[list] = None,
         ax: Optional[plt.Axes] = None, show_sink: bool = False) -> Optional[Tuple[plt.Figure, plt.Axes]]:

    cmap = ListedColormap(['k', 'darkgrey', 'lightgrey', 'w'], name='bwg')
    if show_sink:
        dat, result, xy_labels = add_sink_data(bam, results, copy(xy_labels))
    else:
        dat = np.array(bam.bin_acc, dtype=float, copy=True)
        result = copy(results)

    dat[np.diag_indices(dat.shape[0])] = 0.3
    dat[np.triu_indices(dat.shape[0], k=1)] = 0.6
    if ax is None:
        fig, axis = plt.subplots(figsize=(size, size / 1.5))
    else:
        axis = ax
    axis.imshow(dat, cmap=cmap)
    if xy_labels is not None:
        x_ = list(range(len(dat)))
        axis.set_xticks(x_, labels=xy_labels)
        axis.set_yticks(x_, labels=xy_labels)
        axis.tick_params(axis='both', labelsize='large')
    else:
        axis.set_xticks([])
        axis.set_yticks([])
    add_cell_borders(axis, dat.shape[0])
    set_legend(axis)
    if result:
        if top is None:
            top = result._top
        add_results(axis, result, lim=top)
    if ax is None:
        return fig, axis
