"""Plotting"""

import matplotlib.pyplot as plt


def grid():
    """Grid"""
    plt.grid(visible=True, which='major', linestyle='--', alpha=0.4)
    plt.grid(visible=True, which='minor', linestyle=':', alpha=0.3)
