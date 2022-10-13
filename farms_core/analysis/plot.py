"""Plotting"""

import numpy as np
from cycler import cycler
import matplotlib.pyplot as plt


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
