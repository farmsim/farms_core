"""Plotting"""

from cycler import cycler
import matplotlib.pyplot as plt


def grid():
    """Grid"""
    plt.grid(visible=True, which='major', linestyle='--', alpha=0.4)
    plt.grid(visible=True, which='minor', linestyle=':', alpha=0.3)


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
