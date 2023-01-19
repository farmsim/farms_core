"""Plotting"""

import os
from enum import Enum
from typing import List

import numpy as np
from nptyping import NDArray, Shape, Float
from cycler import cycler
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.interpolate import griddata

from .. import pylog

SEABORN = False
try:
    import seaborn as sns
    SEABORN = True
except ImportError as err:
    pylog.warning('Seaborn not installed, using tableau instead')


def plt_farms_style():
    """Matplotlib FARMS sytle"""
    plt_style_options()
    plt_cycle_options()
    plt_latex_options()


def plt_style_options():
    """Style options for plotting"""
    if SEABORN:
        sns.set_theme(
            context='paper',
            style='darkgrid',
            palette='colorblind',
            rc={
                'font.size': 12.0,
                'axes.facecolor': '#EAEAF2',
                'axes.grid': False,
                'axes.spines.left': True,
                'axes.spines.top': True,
                'axes.spines.right': True,
                'axes.spines.bottom': True,
                'axes.edgecolor': '0.8',
                'axes.linewidth': '0.3',
                'grid.color': '0.7',
                'xtick.top': False,
                'xtick.bottom': True,
                'ytick.left': True,
                'ytick.right': False,
            },
        )
    else:
        plt.style.use('tableau-colorblind10')


def plt_cycle_options():
    """Cycle options for plotting"""
    plt.rc('axes', prop_cycle=(
        cycler(linestyle=['-', '--', '-.', ':'])
        * cycler(color=plt.rcParams['axes.prop_cycle'].by_key()['color'])
    ))


def plt_latex_options():
    """Latex options for plotting"""
    plt.rcParams.update({
        'text.usetex': True,
        'font.family': 'serif',
        'font.serif': ['Palatino'],
    })


def grid(visible=True):
    """Grid"""
    plt.grid(visible=visible, which='major', linestyle='--', alpha=0.4)
    plt.grid(visible=visible, which='minor', linestyle=':', alpha=0.3)


def plt_legend_side(n_labels, max_labels_per_row=20):
    """Legend for plotting"""
    plt.legend(
        bbox_to_anchor=(1.05, 1),
        loc='upper left',
        borderaxespad=0,
        ncol=(n_labels-1)//max_labels_per_row+1,
    )


def save_plots(plots, path, extension='pdf', **kwargs):
    """Save plots"""
    for name, fig in plots.items():
        filename = os.path.join(path, f'{name}.{extension}')
        pylog.debug('Saving to %s', filename)
        fig.savefig(filename, format=extension, **kwargs)


def plot2d(results, labels, n_data=300, log=False, cmap='cividis', **kwargs):
    """Plot result in 2D

    results - The results are given as a 2d array of dimensions [N, 3].

    labels - The labels should be a list of three string for the xlabel, the
    ylabel and zlabel (in that order).

    n_data - Represents the number of points used along x and y to draw the plot

    log - Set log to True for logarithmic scale.

    cmap - You can set the color palette with cmap. For example,
    set cmap='nipy_spectral' for high constrast results.

    """
    show_grid = kwargs.pop('show_grid', True)
    xnew = np.linspace(min(results[:, 0]), max(results[:, 0]), n_data)
    ynew = np.linspace(min(results[:, 1]), max(results[:, 1]), n_data)
    results_interp = griddata(
        points=(results[:, 0], results[:, 1]),
        values=results[:, 2],
        xi=tuple(np.meshgrid(xnew, ynew)),
        method=kwargs.pop('method', 'linear'),  # nearest, linear, cubic
    )
    extent = (
        min(xnew), max(xnew),
        min(ynew), max(ynew)
    )
    plot_kwargs = {}
    for key in ['color']:
        if key in kwargs:
            plot_kwargs[key] = kwargs.pop(key)
    default_markersize = mpl.rcParams['lines.markersize']
    plt.plot(
        results[:, 0],
        results[:, 1],
        linestyle=kwargs.pop('linestyle', 'none'),
        marker=kwargs.pop('marker', 'o'),
        markeredgecolor=kwargs.pop('markeredgecolor', (1, 0.5, 0.5, 0.2)),
        markerfacecolor=kwargs.pop('markerfacecolor', 'none'),
        markersize=kwargs.pop('markersize', 1.2*default_markersize),
        **plot_kwargs,
    )
    plt.plot(
        results[:, 0],
        results[:, 1],
        linestyle=kwargs.pop('linestyle2', 'none'),
        marker=kwargs.pop('marker2', 'o'),
        markeredgecolor=kwargs.pop('markeredgecolor2', (1, 0, 0, 0.2)),
        markerfacecolor=kwargs.pop('markerfacecolor2', 'none'),
        markersize=kwargs.pop('markersize2', default_markersize),
        **plot_kwargs,
    )
    imgplot = plt.imshow(
        results_interp,
        extent=extent,
        aspect='auto',
        origin='lower',
        interpolation='none',
        norm=LogNorm() if log else None
    )
    grid(show_grid)
    if cmap is not None:
        imgplot.set_cmap(cmap)
    plt.xlabel(labels[0])
    plt.ylabel(labels[1])
    axis = plt.gca()
    divider = make_axes_locatable(axis)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cbar = plt.colorbar(imgplot, cax=cax)
    cbar.set_label(labels[2])
    assert not kwargs, kwargs


class MatrixLineType(Enum):
    """Matrix line type"""
    ROW = 0
    COLUMN = 1


class MatrixLine:
    """Matrix line"""

    def __init__(self, line_type, index, **kwargs):
        super().__init__()
        self.line_type: MatrixLineType = line_type
        self.index: int = index
        self.kwargs = kwargs

    @classmethod
    def row(cls, *args, **kwargs):
        """Matrix row line"""
        return cls(line_type=MatrixLineType.ROW, *args, **kwargs)

    @classmethod
    def column(cls, *args, **kwargs):
        """Matrix column line"""
        return cls(line_type=MatrixLineType.COLUMN, *args, **kwargs)


def plot_matrix(
        matrix: NDArray[Shape['Any, Any'], Float],
        fig_name: str,
        labels: List[List[str]],
        clabel: str,
        **kwargs,
):
    """Plot matrix"""
    lines = kwargs.pop('lines', [])
    xlabel = kwargs.pop('xlabel', '')
    ylabel = kwargs.pop('ylabel', '')
    reduce_x = kwargs.pop('line_y', [])
    reduce_y = kwargs.pop('line_x', [])
    reduce_x = kwargs.pop('reduce_x', False)
    reduce_y = kwargs.pop('reduce_y', False)
    xtwin = kwargs.pop('xtwin', [])
    ytwin = kwargs.pop('ytwin', [])
    shape = np.shape(matrix)
    row_map = {i: i for i in range(shape[0])}
    col_map = {i: i for i in range(shape[1])}
    row_map[shape[0]] = shape[0]-1
    col_map[shape[1]] = shape[1]-1
    if reduce_x or reduce_y:
        labels = labels[:]
        matrix_nans = np.isnan(matrix)
        if reduce_y:
            rows = [not all(matrix_nans[i, :]) for i in range(shape[0])]
            j = -1
            for i in range(shape[0]):
                if rows[i]:
                    j += 1
                row_map[i] = j
            row_map[shape[0]] = j
            matrix = matrix[rows, :]
            labels[0] = np.array(labels[0])[rows]
        if reduce_x:
            cols = [not all(matrix_nans[:, i]) for i in range(shape[1])]
            j = -1
            for i in range(shape[1]):
                if cols[i]:
                    j += 1
                col_map[i] = j
            col_map[shape[1]] = j
            matrix = matrix[:, cols]
            labels[1] = np.array(labels[1])[cols]
        shape = np.shape(matrix)
    figsize = (0.15*shape[1]+0.5, 0.15*shape[0])
    figure = plt.figure(fig_name, figsize=figsize)
    axes = plt.gca()
    if 0 in shape:
        return figure
    assert shape[0] == len(labels[0]), f'{shape[0]=} == {len(labels[0])=}'
    assert shape[1] == len(labels[1]), f'{shape[1]=} == {len(labels[1])=}'

    # Plot matrix
    # ims = axes.matshow(matrix, **kwargs)
    ims = axes.imshow(matrix, **kwargs)
    axes.autoscale(False)
    axes.set_xticks(np.arange(shape[1]))
    axes.set_yticks(np.arange(shape[0]))
    axes.set_xticklabels(labels[1], rotation='vertical')
    axes.set_yticklabels(labels[0], rotation='horizontal')

    # Range limits
    axes.set_xlim(left=-0.5, right=shape[1]-0.5)
    axes.set_ylim(top=-0.5, bottom=shape[0]-0.5)

    # Plot lines
    for line in lines:
        assert isinstance(line, MatrixLine), f'{line=} if not {MatrixLine}'
        assert isinstance(line.line_type, MatrixLineType), (
            f'{line.line_type=} if not {MatrixLineType}'
        )
        if line.line_type == MatrixLineType.ROW:
            pos = line.index + row_map[round(line.index)] - round(line.index)
            axes.plot(
                [-0.5, shape[1]-0.5],
                [pos, pos],
                **line.kwargs,
            )
        else:
            pos = line.index + col_map[round(line.index)] - round(line.index)
            axes.plot(
                [pos, pos],
                [-0.5, shape[0]-0.5],
                **line.kwargs,
            )

    # X-axis twin
    if xtwin:
        twinx = ims.axes.twiny()
        twinx.autoscale(False)
        twinx.set_xlim(*axes.get_xlim())
        twinx.set_ylim(*axes.get_ylim())
        twinx.xaxis.set_ticks_position('bottom')
        twinx.xaxis.set_label_position('bottom')
        twinx.spines.bottom.set_position(('axes', 0.0))
        xticks_major = [-0.5] + [
            0.5 + col_map[tick_data[1][1]]
            for tick_data in xtwin
        ]
        xticks_minor = [
            0.5*(col_map[tick_data[1][0]]+col_map[tick_data[1][1]]+1)
            for tick_data in xtwin
        ]
        xticklabels = [tick_data[0] for tick_data in xtwin]
        twinx.set_xticks(xticks_major)
        twinx.set_xticks(xticks_minor, minor=True)
        twinx.set_xticklabels(xticklabels, minor=True)
        twinx.tick_params(axis='x', which='minor', length=0)
        for xlabel_i in twinx.get_xticklabels():
            xlabel_i.set_visible(False)

    pylog.debug('Column map:\n%s', col_map)

    # Y-axis twin
    if ytwin:
        twiny = ims.axes.twinx()
        twiny.autoscale(False)
        twiny.set_xlim(*axes.get_xlim())
        twiny.set_ylim(*axes.get_ylim())
        twiny.yaxis.set_ticks_position('right')
        twiny.yaxis.set_label_position('right')
        twiny.spines.right.set_position(('axes', 1.0))
        yticks_major = [-0.5] + [
            0.5 + row_map[tick_data[1][1]]
            for tick_data in ytwin
        ]
        yticks_minor = [
            0.5*(row_map[tick_data[1][0]]+row_map[tick_data[1][1]]+1)
            for tick_data in ytwin
        ]
        yticklabels = [tick_data[0] for tick_data in ytwin]
        twiny.set_yticks(yticks_major)
        twiny.set_yticks(yticks_minor, minor=True)
        twiny.set_yticklabels(yticklabels, minor=True, rotation='vertical')
        twiny.tick_params(axis='y', which='minor', length=0)
        for ylabel_i in twiny.get_yticklabels():
            ylabel_i.set_visible(False)

    # Axes
    axes.xaxis.set_ticks_position('top')
    axes.xaxis.set_label_position('top')
    axes.yaxis.set_ticks_position('left')
    axes.yaxis.set_label_position('left')

    # Labels
    if xlabel:
        axes.set_xlabel(xlabel)
        axes.xaxis.set_label_position('top')
    if ylabel:
        axes.set_ylabel(ylabel)
        axes.yaxis.set_label_position('left')

    # Grid
    axes.set_xticks(np.arange(shape[1]+1)-0.5, minor=True)
    axes.set_yticks(np.arange(shape[0]+1)-0.5, minor=True)
    axes.grid(which='minor', color='w', linestyle='-', linewidth=1)
    axes.tick_params(which='minor', top=False, left=False)
    axes.tick_params(which='both', bottom=False, right=False)

    # Colorbar
    cbar = figure.colorbar(ims)
    cbar.set_label(clabel)
    # divider = make_axes_locatable(axis)
    # dax = divider.append_axes('right', size='5%', pad=0.05)
    # cbar = plt.colorbar(cax, cax=dax)
    # cbar.set_label(clabel)

    return figure


def colorgraph(
        data,
        labels=None,
        n_pixel_x=1,
        n_pixel_y=1,
        gap=10,
        **kwargs
):
    """Plot color graph

    data: 2D array of shape (n_elements, n_iterations)
    labels: Labels of each bar along the y-axis (default = range values)
    cmap: Color map
    n_pixel_x: Number of x pixels to color (choose 1 for time plot)
    n_pixel_y: Number of y pixels to color
    gap: Number of gap pixels

    """

    # Params
    xlabel = kwargs.pop('xlabel', None)
    ylabel = kwargs.pop('ylabel', None)
    clabel = kwargs.pop('clabel', None)
    x_extent = kwargs.pop('x_extent', [0, len(data[0])-1])
    show_grid = kwargs.pop('show_grid', True)
    if 'extent' not in kwargs:
        kwargs['extent'] = [x_extent[0], x_extent[1], len(labels), 0]
    if 'interpolation' not in kwargs:
        kwargs['interpolation'] = 'none'
    n_elements, n_iters = np.shape(data)
    if labels is None:
        labels=range(n_elements)
    assert len(labels) == n_elements, f'{len(labels)} != {n_elements}'

    # Data
    arr = np.insert(data, range(1, n_elements+1), np.zeros(n_iters), axis=0)
    arr = np.repeat(arr, n_pixel_x, axis=1)
    arr = np.repeat(arr, np.tile([n_pixel_y, gap], n_elements), axis=0)

    # Plot
    axis = plt.gca()
    imgplot = plt.imshow(arr, **kwargs)
    grid(show_grid)
    axis.yaxis.grid(visible=False)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.yticks(
        ticks=[i+0.5*n_pixel_y/(n_pixel_y+gap) for i in range(n_elements)],
        labels=labels,
    )
    divider = make_axes_locatable(axis)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cbar = plt.colorbar(imgplot, cax=cax)
    cbar.set_label(clabel)
