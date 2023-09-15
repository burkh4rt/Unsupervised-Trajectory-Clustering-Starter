#!/usr/bin/env python3

"""
Utility functions
"""

import pathlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.metrics as skl_mets

pd.options.display.width = 79
pd.options.display.max_columns = 1000
pd.options.display.max_colwidth = 79
pd.options.display.float_format = "{:,.3f}".format

plt.rcParams["figure.constrained_layout.use"] = True


def plot_matrix(
    mat: np.array,
    *,
    show_colorbar: bool = False,
    show_labels: bool = True,
    rotate_xlabels: bool = True,
    xticks: list = None,
    xlabel: str = None,
    yticks: list = None,
    ylabel: str = None,
    title: str = None,
    fmt_str: str = "{:.2f}",
    figsize: tuple = (6.4, 4.8),
    savename: pathlib.Path | str = None,
    show_figure: bool = False,
):
    mat = np.atleast_2d(mat)
    fig, ax = plt.subplots(layout="constrained", figsize=figsize)
    im = ax.matshow(mat, cmap="cividis")
    if show_colorbar:
        ax.figure.colorbar(im, ax=ax)
    if xticks:
        ax.xaxis.set_label_position("top")
        ax.set_xticks(np.arange(len(xticks)), labels=xticks)
        if rotate_xlabels:
            plt.setp(
                ax.get_xticklabels(),
                rotation=-30,
                ha="right",
                rotation_mode="anchor",
            )
    if yticks is not None:
        ax.set_yticks(np.arange(len(yticks)), labels=yticks)
    if title is not None:
        plt.title(title)
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)
    if show_labels:
        # thanks, @geology.beer
        for (_i, _j), _z in np.ndenumerate(mat):
            ax.text(
                _j,
                _i,
                fmt_str.format(_z),
                ha="center",
                va="center",
                c="black" if _z > np.mean(im.get_clim()) else "white",
            )

    plt.tick_params(bottom=False)
    _m, _n = mat.shape
    if _m == 1:
        plt.tick_params(left=False, labelleft=False, bottom=False)

    if savename is not None:
        plt.savefig(savename)
    if show_figure:
        plt.show()
