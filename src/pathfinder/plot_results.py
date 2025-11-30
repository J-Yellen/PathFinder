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
from .dfs import WHDFS, HDFS
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


def add_results(
        ax: Axes,
        res: Results,
        lim: int,
        plot_sorted: bool = False,
        bam: Optional[BinaryAcceptance] = None
) -> None:
    """
    Plot result paths on the axes.

    Args:
        ax: Matplotlib axes object to plot on.
        res: Results object containing paths to visualise.
        lim: Maximum number of paths to plot.
        plot_sorted: If True, plot paths in sorted index space. If False, plot in original space.
        bam: BinaryAcceptance object (unused, kept for API compatibility).

    Returns:
        None
    """
    # Import here to avoid circular import
    from pathfinder.dfs import HDFS

    # Use sorted paths when plot_sorted=True
    if plot_sorted:
        if isinstance(res, WHDFS) and hasattr(res, 'get_sorted_paths'):
            paths_to_plot = res.get_sorted_paths()[:lim:]
        elif isinstance(res, (HDFS, WHDFS)):
            # For HDFS or WHDFS, get paths and remap to sorted space if index_map exists
            if res.bam._index_map is not None:
                # Create reverse mapping (original -> sorted)
                reverse_map = {v: k for k, v in enumerate(res.bam._index_map)}
                paths_to_plot = [sorted([reverse_map[i] for i in path]) for path in res.get_paths[:lim:]]
            else:
                paths_to_plot = res.get_paths[:lim:]
        else:
            # Plain Results object - just use get_paths
            paths_to_plot = res.get_paths[:lim:]
    else:
        # Default: plot in original index space for both HDFS and WHDFS
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


def plot(bam: Optional[BinaryAcceptance] = None, results: Optional[Union[WHDFS, HDFS, Results]] = None,
         top: Optional[int] = None, size: int = 16, axis_labels: bool = True, xy_labels: Optional[List[str]] = None,
         ax: Optional[Axes] = None, show_sink: bool = False, plot_sorted: bool = False,
         highlight_top_path: bool = False) -> Tuple[Figure, Axes]:
    """
    Visualise Binary Acceptance Matrix with optional result paths overlaid.

    Creates a heatmap visualisation of the BAM where:
    - White cells indicate acceptable pairwise relations (below threshold)
    - Black cells indicate unacceptable relations (above threshold)
    - Grey diagonal represents self-relations
    - Coloured lines show result paths

    Args:
        bam: BinaryAcceptance object containing the matrix to visualise. If None and
             results is an HDFS/WHDFS object, extracts BAM from results.bam.
        results: Optional Results object containing paths to overlay on the plot.
                 Can be HDFS, WHDFS, or Results object.
        top: Maximum number of result paths to display. If None, uses results._top.
        size: Figure width in inches. Height is size/1.5. Default is 16.
        axis_labels: If True, use labels from BinaryAcceptance object (respects sorting order).
                    Ignored if xy_labels is provided. Default True.
        xy_labels: Optional labels for axes. If provided, overrides axis_labels. If None
                   and axis_labels=False, no labels shown.
        ax: Optional existing Axes to plot on. If None, creates new figure.
        show_sink: If True, extends matrix to show dummy sink/target node. Default False.
        plot_sorted: If True, plot paths in sorted index space (weight-ordered).
                     If False (default), plot in original index space for visual comparison.
        highlight_top_path: If True, highlight rows/columns corresponding to the best path.
                            Default False.

    Returns:
        Tuple of (Figure, Axes). If ax was provided, Figure is retrieved from the axes.

    Example:
        >>> # Simplest usage - just pass results object
        >>> whdfs = WHDFS(bam, top=5)
        >>> whdfs.find_paths()
        >>> fig, ax = plot(whdfs, highlight_top_path=True)
        >>>
        >>> # Or explicitly provide both
        >>> fig, ax = plot(bam, results, size=12)
        >>>
        >>> # Using labels from BinaryAcceptance object:
        >>> fig, ax = plot(results, axis_labels=True)
        >>>
        >>> # For weight-ordered visualisation:
        >>> fig, ax = plot(bam, results, plot_sorted=True)
    """
    # Handle flexible parameter order: if bam looks like Results, swap parameters
    if bam is not None and hasattr(bam, 'bam') and not isinstance(bam, BinaryAcceptance):
        # User passed Results as first arg, swap to results parameter
        if results is None:
            results = bam  # type: ignore[assignment]
            bam = results.bam  # type: ignore[attr-defined]
        else:
            raise ValueError("Ambiguous parameters: bam appears to be a Results object but results is also provided")

    # Extract BAM from results if not provided
    if bam is None and results is not None:
        if hasattr(results, 'bam'):
            bam = results.bam  # type: ignore[attr-defined]
        else:
            raise ValueError("bam parameter is required when results object has no 'bam' attribute")

    if bam is None:
        raise ValueError("Either bam parameter or results with bam attribute must be provided")

    cmap = ListedColormap(['k', 'darkgrey', 'lightgrey', 'w'], name='bwg')

    # Handle label selection: xy_labels overrides axis_labels
    if xy_labels is None and axis_labels and bam.labels is not None:
        xy_labels = copy(bam.labels)

    if show_sink and results is not None:
        dat, result, xy_labels = add_sink_data(bam, results, copy(xy_labels))
    else:
        dat = np.array(bam.bin_acc, dtype=float, copy=True)
        result = copy(results)

    # If plot_sorted=False and BAM has been sorted, we need to UNSORT it for display
    # so paths (which are in original indices) align correctly with the matrix
    if not plot_sorted and bam._index_map is not None:
        # Create reverse mapping to unsort the matrix
        reverse_indices = np.argsort(bam._index_map)
        dat = dat[reverse_indices, :][:, reverse_indices]
        # Also unsort labels if they exist
        if xy_labels is not None:
            xy_labels = [xy_labels[i] for i in reverse_indices]

    dat[np.diag_indices(dat.shape[0])] = 0.3
    dat[np.triu_indices(dat.shape[0], k=1)] = 0.6

    # Create new figure/axes or use provided axes
    if ax is None:
        fig, axis = plt.subplots(figsize=(size, size / 1.5))
    else:
        axis = ax
        fig = cast(Figure, axis.get_figure())

    axis.imshow(dat, cmap=cmap)

    # Calculate top_path if highlighting is requested
    top_path = None
    if highlight_top_path and result and len(result.get_paths) > 0:
        # Get the top path in the correct index space for the displayed matrix
        if plot_sorted:
            # If plotting in sorted space, get sorted paths
            if isinstance(result, WHDFS) and hasattr(result, 'get_sorted_paths'):
                top_path = result.get_sorted_paths()[0]
            elif bam._index_map is not None:
                # Map to sorted space
                reverse_map = {v: k for k, v in enumerate(bam._index_map)}
                top_path = sorted([reverse_map[i] for i in result.get_paths[0]])
            else:
                top_path = result.get_paths[0]
        else:
            # Use original space paths
            top_path = result.get_paths[0]

        # Draw semi-transparent rectangles to highlight rows (up to and including diagonal)
        for idx in top_path:
            # Highlight row from start through the diagonal box (lower triangle + diagonal)
            axis.axhspan(idx - 0.5, idx + 0.5, xmin=0.0,
                         xmax=(idx + 1.0) / dat.shape[0], alpha=0.4, color='green', zorder=1)

    if xy_labels is not None:
        x_ = list(range(len(dat)))
        axis.set_xticks(x_, labels=xy_labels, rotation='vertical')
        axis.set_yticks(x_, labels=xy_labels, rotation='horizontal')
        axis.tick_params(axis='both', labelsize='large')

        # Color labels in top path dark green if highlighting is enabled
        if top_path is not None:
            # Set color for highlighted labels
            for label in axis.get_xticklabels():
                if int(label.get_position()[0]) in top_path:
                    label.set_color('darkgreen')
                    label.set_fontweight('bold')
            for label in axis.get_yticklabels():
                if int(label.get_position()[1]) in top_path:
                    label.set_color('darkgreen')
                    label.set_fontweight('bold')
    else:
        axis.set_xticks([])
        axis.set_yticks([])
    add_cell_borders(axis, dat.shape[0])
    set_legend(axis)
    if result:
        if top is None:
            top = result._top if result._top is not None else len(result.get_paths)
        add_results(axis, result, lim=top, plot_sorted=plot_sorted, bam=bam)

    return fig, axis
