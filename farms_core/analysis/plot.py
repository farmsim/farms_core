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
    if reduce_x or reduce_y:
        labels = labels[:]
        shape = np.shape(matrix)
        matrix_nans = np.isnan(matrix)
        if reduce_y:
            rows = [
                not all(matrix_nans[i, :])
                for i in range(shape[0])
            ]
            matrix = matrix[rows, :]
            labels[0] = np.array(labels[0])[rows]
        if reduce_x:
            cols = [
                not all(matrix_nans[:, i])
                for i in range(shape[1])
            ]
            matrix = matrix[:, cols]
            labels[1] = np.array(labels[1])[cols]
    shape = np.shape(matrix)
    figure = plt.figure(fig_name, figsize=(0.15*shape[1]+2, 0.15*shape[0]))
    axis = plt.gca()
    if 0 in shape:
        return figure
    assert shape[0] == len(labels[0])
    assert shape[1] == len(labels[1])

    # Plot matrix
    cax = axis.matshow(matrix, **kwargs)
    axis.set_xticks(range(shape[1]))
    axis.set_yticks(range(shape[0]))
    axis.set_xticklabels(labels[1], rotation='vertical')
    axis.set_yticklabels(labels[0], rotation='horizontal')

    # Labels
    if xlabel:
        axis.set_xlabel(xlabel)
        axis.xaxis.set_label_position('top')
    if ylabel:
        axis.set_ylabel(ylabel)

    # Plot lines
    axis.autoscale(False)
    for line in lines:
        assert isinstance(line, MatrixLine), f'{line=} if not {MatrixLine}'
        assert isinstance(line.line_type, MatrixLineType), (
            f'{line.line_type=} if not {MatrixLineType}'
        )
        if line.line_type == MatrixLineType.ROW:
            axis.plot([-0.5, shape[1]], [line.index, line.index], **line.kwargs)
        else:
            axis.plot([line.index, line.index], [-0.5, shape[0]], **line.kwargs)

    # Colorbar
    cbar = figure.colorbar(cax)
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
