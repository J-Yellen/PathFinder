#!/usr/bin/env python3
"""
#####################################
# Part of the PATH FINDER  Module   #
# Author J.Yellen                   #
#####################################
"""
import numpy as np
from copy import copy
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib import font_manager
from .matrix_handler import BinaryAcceptance
from .result import Results
from .dfs import WHDFS
from typing import Optional, Tuple, List, Union, Set, Dict, Sequence, cast

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


def set_legend(ax: Axes) -> None:
    """
    Add legend to axes showing BAM matrix interpretation.

    Args:
        ax: Matplotlib axes object to add legend to.

    Returns:
        None
    """
    legend_elements = [Rectangle((0.0, 0.0), 1, 2, facecolor='w', edgecolor='k', label=r'$\rho_{ij} < T$'),
                       Rectangle((0.0, 0.0), 1, 2, facecolor='k', edgecolor='k', label=r'$\rho_{ij} \geq T$'),
                       Rectangle((0.0, 0.0), 1, 2, facecolor='darkgrey', edgecolor='k', label=r'$\rho_{ii}$')
                       ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize='30', framealpha=1)


def add_cell_borders(ax: Axes, dim: int) -> None:
    """
    Add cell borders to BAM matrix visualisation.

    Args:
        ax: Matplotlib axes object to add borders to.
        dim: Dimension of the matrix (number of rows/columns).

    Returns:
        None
    """
    for i in range(0, dim):
        ax.axhline(y=i + 0.5, xmin=0, xmax=(i + 2) / dim, linewidth=2, color="k")
        ax.axvline(x=i + 0.5, ymin=0, ymax=(dim - i) / dim, linewidth=2, color="k")


def format_path(path: List[int]) -> np.ndarray:
    """
    Convert path indices into x,y coordinates for plotting on BAM matrix.

    Creates stepped line coordinates that follow the matrix structure,
    moving horizontally then vertically between path elements.

    Args:
        path: Sorted list of integer indices representing a path through the matrix.

    Returns:
        2D numpy array with shape (2, N) containing [xval, yval] coordinates.
        First row contains x-coordinates, second row contains y-coordinates.

    Example:
        >>> format_path([0, 1, 2])
        array([[[0, 0], [0, 1], [1, 1], [1, 2]],
               [[0, 1], [1, 1], [1, 2], [2, 2]]])
    """
    xval, yval = [], []
    if len(path) > 1:
        for i, j in zip(path[:-1:], path[1::]):
            xval += [[i, i], [i, j]]
            yval += [[i, j], [j, j]]
    else:
        xval = [path]
        yval = [path]
    return np.array([xval, yval])


def make_path(
        paths: Sequence[Union[List[int], Set[int]]],
        shift: bool = True
) -> Dict[int, Dict[str, Union[List[float], np.ndarray]]]:
    """
    Convert multiple paths into plotting coordinates with colour assignments.

    Args:
        paths: Sequence of paths, where each path is either a list or set of integer indices.
        shift: If True, slightly offset each path's coordinates to prevent overlapping
               when multiple paths share edges. Default is True.

    Returns:
        Dictionary mapping path index to path plotting data. Each path data dict contains:
            - 'x': List of x-coordinates for plotting
            - 'y': List of y-coordinates for plotting
            - 'color': RGBA colour array for the path
    """
    shift_i = 1. / (len(paths) + 1)
    lshft = 0.5
    c_vals = np.linspace(0.0, 1, len(paths))
    cmap = plt.colormaps.get_cmap('rainbow')
    color = cmap(c_vals)
    ret = {}
    for i, p in enumerate(paths):
        ret[i] = {}
        posx, posy = [], []
        if shift:
            lshft = (i + 1) * shift_i
        # Convert to list if set
        path_list = sorted(p) if isinstance(p, set) else p
        pth = format_path(path_list)
        for j, k in zip(pth[0], pth[1]):
            xy = np.array([j, k])
            posx += list(xy[0] + lshft - 0.5)
            posy += list(xy[1] + lshft - 0.5)
        ret[i]['x'] = posx
        ret[i]['y'] = posy
        loc = abs(c_vals - (1 - i / len(paths))).argmin()
        ret[i]['color'] = color[loc]
    return ret


def add_results(ax: Axes, res: Results, lim: int) -> None:
    """
    Plot result paths on the axes.

    Args:
        ax: Matplotlib axes object to plot on.
        res: Results object containing paths to visualise.
        lim: Maximum number of paths to plot.

    Returns:
        None
    """
    # For WHDFS with auto_sort, use sorted paths to match the sorted BAM
    if isinstance(res, WHDFS) and hasattr(res, 'get_sorted_paths'):
        paths_to_plot = res.get_sorted_paths()[:lim:]
    else:
        paths_to_plot = res.get_paths[:lim:]

    for i, item in make_path(paths_to_plot, shift=True).items():
        if len(item['x']) > 1:
            ax.plot(item['x'], item['y'], lw=3, color=item['color'], linestyle='-',
                    label=f"{res.get_weights[i]:.2g}")


def add_sink_data(
        bam: BinaryAcceptance,
        result: Results,
        xy_labels: Optional[List[str]] = None
) -> Tuple[np.ndarray, Results, List[str]]:
    """
    Extend BAM matrix and results to include dummy sink/target node.

    Args:
        bam: BinaryAcceptance object containing the matrix.
        result: Results object containing paths.
        xy_labels: Optional list of axis labels. If None, generates numeric labels.

    Returns:
        Tuple containing:
            - Extended matrix data with sink node
            - Updated Results object with sink node added to paths
            - Extended labels list with 'Sink' label
    """
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
         size: int = 16, xy_labels: Optional[List[str]] = None,
         ax: Optional[Axes] = None, show_sink: bool = False) -> Tuple[Figure, Axes]:
    """
    Visualise Binary Acceptance Matrix with optional result paths overlaid.

    Creates a heatmap visualisation of the BAM where:
    - White cells indicate acceptable pairwise relations (below threshold)
    - Black cells indicate unacceptable relations (above threshold)
    - Grey diagonal represents self-relations
    - Coloured lines show result paths

    Args:
        bam: BinaryAcceptance object containing the matrix to visualise.
        results: Optional Results object containing paths to overlay on the plot.
        top: Maximum number of result paths to display. If None, uses results._top.
        size: Figure width in inches. Height is size/1.5. Default is 16.
        xy_labels: Optional labels for axes. If None, no labels shown.
        ax: Optional existing Axes to plot on. If None, creates new figure.
        show_sink: If True, extends matrix to show dummy sink/target node. Default False.

    Returns:
        Tuple of (Figure, Axes). If ax was provided, Figure is retrieved from the axes.

    Example:
        >>> bam = BinaryAcceptance(matrix, weights=weights, threshold=0.5)
        >>> results = HDFS(bam, top=5).find_paths()
        >>> fig, ax = plot(bam, results, size=12)
    """
    cmap = ListedColormap(['k', 'darkgrey', 'lightgrey', 'w'], name='bwg')
    if show_sink and results is not None:
        dat, result, xy_labels = add_sink_data(bam, results, copy(xy_labels))
    else:
        dat = np.array(bam.bin_acc, dtype=float, copy=True)
        result = copy(results)

    dat[np.diag_indices(dat.shape[0])] = 0.3
    dat[np.triu_indices(dat.shape[0], k=1)] = 0.6

    # Create new figure/axes or use provided axes
    if ax is None:
        fig, axis = plt.subplots(figsize=(size, size / 1.5))
    else:
        axis = ax
        fig = cast(Figure, axis.get_figure())

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

    return fig, axis
