"""Plotting"""

import os
import numpy as np
from cycler import cycler
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.interpolate import griddata
from .. import pylog


def plt_farms_style():
    """Matplotlib FARMS sytle"""
    plt_colorblind_options()
    plt_cycle_options()
    plt_latex_options()


def plt_colorblind_options():
    """Colorblind options for plotting"""
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


def grid():
    """Grid"""
    plt.grid(visible=True, which='major', linestyle='--', alpha=0.4)
    plt.grid(visible=True, which='minor', linestyle=':', alpha=0.3)


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
    xnew = np.linspace(min(results[:, 0]), max(results[:, 0]), n_data)
    ynew = np.linspace(min(results[:, 1]), max(results[:, 1]), n_data)
    grid_x, grid_y = np.meshgrid(xnew, ynew)
    results_interp = griddata(
        (results[:, 0], results[:, 1]), results[:, 2],
        (grid_x, grid_y),
        method='linear',  # nearest, cubic
    )
    extent = (
        min(xnew), max(xnew),
        min(ynew), max(ynew)
    )
    plot_kwargs = {}
    for key in ['color']:
        if key in kwargs:
            plot_kwargs[key] = kwargs.pop(key)
    plt.plot(
        results[:, 0],
        results[:, 1],
        linestyle=kwargs.pop('linestyle', 'none'),
        marker=kwargs.pop('marker', 'o'),
        markeredgecolor=kwargs.pop('markeredgecolor', (1, 0, 0, 0.5)),
        markerfacecolor=kwargs.pop('markerfacecolor', 'none'),
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
    if cmap is not None:
        imgplot.set_cmap(cmap)
    plt.xlabel(labels[0])
    plt.ylabel(labels[1])
    cbar = plt.colorbar()
    cbar.set_label(labels[2])
    assert not kwargs, kwargs


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
    xlabel = kwargs.pop('xlabel', None)
    ylabel = kwargs.pop('ylabel', None)
    clabel = kwargs.pop('clabel', None)
    x_extent = kwargs.pop('x_extent', [0, len(data[0])-1])
    if 'extent' not in kwargs:
        kwargs['extent'] = [x_extent[0], x_extent[1], len(labels), 0]
    if 'interpolation' not in kwargs:
        kwargs['interpolation'] = 'none'
    n_elements, n_iters = np.shape(data)
    if labels is None:
        labels=range(n_elements)
    assert len(labels) == n_elements, f'{len(labels)} != {n_elements}'
    arr = np.insert(data, range(1, n_elements+1), np.zeros(n_iters), axis=0)
    arr = np.repeat(arr, n_pixel_x, axis=1)
    arr = np.repeat(arr, np.tile([n_pixel_y, gap], n_elements), axis=0)
    plt.imshow(arr, **kwargs)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.yticks(
        ticks=[i+0.5*n_pixel_y/(n_pixel_y+gap) for i in range(n_elements)],
        labels=labels,
    )
    cbar = plt.colorbar()
    cbar.set_label(clabel)
