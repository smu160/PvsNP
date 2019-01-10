"""This module contains wrapper functions for plotting and visualizing data.

    @author: Saveliy Yusufov, Columbia University, sy2685@columbia.edu
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns
from scipy.ndimage.filters import gaussian_filter

def generate_heatmap(x, y, sigma=2, **kwargs):
    """Generates a heatmap for plotting.

    Wrapper function meant for generating heatmap using the gaussian_filter
    function implemented in scipy, as well as NumPy's histogram2d function.

    Sources:
        https://docs.scipy.org/doc/numpy/reference/generated/numpy.histogram2d.html
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.gaussian_filter.html

    Args:
        x: array_like, shape (N,)
            An array containing the x coordinates of the points to be
            histogrammed.

        y: array_like, shape (N,)
            An array containing the y coordinates of the points to be
            histogrammed.

        bins: int or array_like or [int, int], optional, default: (50, 50)

        sigma: scalar, optional, default: 2
            Standard deviation for Gaussian kernel.

        weights : array_like, shape(N,), optional, default: None
            An array of values ``w_i`` weighing each sample ``(x_i, y_i)``.
            Weights are normalized to 1 if `normed` is True. If `normed` is
            False, the values of the returned histogram are equal to the sum of
            the weights belonging to the samples falling into each bin.

    Returns:
        heatmap: ndarray
            Returned array of same shape as input. We return the transpose of
            the array in order to preserve the view.

        extent: scalars (left, right, bottom, top)
            The bounding box in data coordinates that the image will fill. The
            image is stretched individually along x and y to fill the box.
            Source: https://matplotlib.org/api/_as_gen/matplotlib.pyplot.imshow.html
    """
    bins = kwargs.get("bins", (50, 50))
    weights = kwargs.get("weights", None)

    heatmap, xedges, yedges = np.histogram2d(x, y, bins=bins, weights=weights)
    heatmap = gaussian_filter(heatmap, sigma=sigma)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    return heatmap.T, extent

def plot_heatmap(x, y, sigma=2, **kwargs):
    """Plots a heatmap.

    Wrapper function for matplotlib and generate_heatmap that plots the actual
    generated heatmap.

    Args:
        x: array_like, shape (N,)
            An array containing the x coordinates of the points to be
            histogrammed.

        y: array_like, shape (N,)
            An array containing the y coordinates of the points to be
            histogrammed.

        bins: int or array_like or [int, int], optional, default: (50, 50)

        sigma: scalar, optional; default: 2
            Standard deviation for Gaussian kernel.

        weights: array_like, shape(N,), optional, default: None
            An array of values ``w_i`` weighing each sample ``(x_i, y_i)``.
            Weights are normalized to 1 if `normed` is True. If `normed` is
            False, the values of the returned histogram are equal to the sum of
            the weights belonging to the samples falling into each bin.

        bounds: tuple, optional, default: None
            If a boundary tuple is provided, the x-axis will be set as follows:
            [bounds[0], bounds[1]]. Similarly, the y-axis will be set as:
            [bounds[0], bounds[1]].

        cmap: matplotlib.colors.LinearSegmentedColormap, optional, default: plt.cm.jet
            The colormap to use for plotting the heatmap.

        figsize: tuple, optional, default: (10, 10)
            The size of the heatmap plot.

        title: str, optional, default: 'Title Goes Here'
            The title of the heatmap plot.
            Note: If title is provided, title will be used as the name of
            the file when the figure is saved.

        dpi: int, optional, default: 600
            The amount of dots per inch to use when saving the figure. In
            accordance with Nature's guidelines, the default is 600.
            Source: https://www.nature.com/nature/for-authors/final-submission

        savefig: bool, optional, default: False
            When True, the plotted heatmap will be saved to the current working
            directory in pdf (IAW Nature's guidelines) format.
            Source: https://www.nature.com/nature/for-authors/final-submission
    """
    cmap = kwargs.get("cmap", cm.jet)
    title = kwargs.get("title", "Title Goes Here")
    bins = kwargs.get("bins", (50, 50))
    weights = kwargs.get("weights", None)
    figsize = kwargs.get("figsize", (10, 10))
    bounds = kwargs.get("bounds", None)

    # Set user-defined x-axis and y-axis boundaries by appending them to x and y
    if bounds:
        x = x.copy()
        y = y.copy()
        x.loc[len(x)] = bounds[0]
        x.loc[len(x)] = bounds[1]
        y.loc[len(y)] = bounds[0]
        y.loc[len(y)] = bounds[1]
        if not weights is None:
            weights = weights.copy()
            weights.loc[len(weights)] = 0
            weights.loc[len(weights)] = 0

    _ = plt.figure(figsize=figsize)
    heatmap, extent = generate_heatmap(x, y, sigma, bins=bins, weights=weights)
    plt.imshow(heatmap, origin="lower", extent=extent, cmap=cmap)

    if title:
        plt.title(title)

    dpi = kwargs.get("dpi", 600)
    savefig = kwargs.get("savefig", False)
    if savefig:
        if title:
            filename = title
        else:
            filename = "my_smoothed_heatmap"

        plt.savefig("{}.pdf".format(filename), dpi=dpi)

        
def abline(slope, intercept):
    """
    Plot a line from slope and intercept
    Inputs:
        slope: float
        intercept: float
    """
    axes = plt.gca()
    x_vals = np.array(axes.get_xlim())
    y_vals = intercept + slope * x_vals
    plt.plot(x_vals, y_vals, '--')
    
    
def pie_chart(sizes, *labels, **kwargs):
    """Wrapper method for matplotlib's pie chart.

    The slices will be ordered and plotted counter-clockwise.

    Args:
        sizes: list
            A list of the sizes of each category.

        labels: str, variable number of args
            The label for each corresponding slice size.

        figsize: tuple, optional, default: (5, 5)
            The size of the figure to be plotted.
    """
    if len(labels) != len(sizes):
        raise ValueError("Length of sizes and amount of labels must be equal.")

    figsize = kwargs.get("figsize", (5, 5))

    _, ax1 = plt.subplots(figsize=figsize)
    ax1.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=90)
    ax1.axis("equal")
    plt.show()

def plot_corr_heatmap(dataframe, **kwargs):
    """Seaborn correlation heatmap wrapper function

    A wrapper function for seaborn to quickly plot a
    correlation heatmap with a lower triangle, only.

    Args:
        dataframe: DataFrame
            A Pandas dataframe to be plotted in the correlation heatmap.

        figsize: tuple, optional, default: (16, 16)
            The size of the heatmap to be plotted.

        title: str, optional, default: None
            The title of the heatmap plot.
            Note: If title is provided, title will be used as the name of
            the file when the figure is saved.

        dpi: int, optional, default: 600
            The amount of dots per inch to use when saving the figure. In
            accordance with Nature's guidelines, the default is 600.
            Source: https://www.nature.com/nature/for-authors/final-submission

        savefig: bool, optional, default: False
            When True, the plotted heatmap will be saved to the current working
            directory in pdf (IAW Nature's guidelines) format.
            Source: https://www.nature.com/nature/for-authors/final-submission
    """
    title = kwargs.get("title", None)

    # Generate a mask for the upper triangle
    mask = np.zeros_like(dataframe.corr(), dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    # Set up the matplotlib figure
    _, _ = plt.subplots(figsize=kwargs.get("figsize", (16, 16)))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(dataframe.corr(), mask=mask, cmap=cmap, vmax=1.0, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})

    if title:
        plt.title(title)

    dpi = kwargs.get("dpi", 600)
    savefig = kwargs.get("savefig", False)
    if savefig:
        if title:
            filename = title
        else:
            filename = "my_seaborn_heatmap"

        plt.savefig("{}.pdf".format(filename), dpi=dpi)

def plot_clustermap(dataframe, **kwargs):
    """Seaborn clustermap wrapper function

    A wrapper function for seaborn to quickly plot a clustermap using the
    "centroid" method to find clusters.

    Args:
        dataframe: DataFrame
            The Pandas dataframe to be plotted in the clustermap.

        figsize: tuple, optional, default: (15, 15)
            The size of the clustermap to be plotted.

        dendrograms: bool, optional, default: True
            If set to False, the dendrograms (row & col) will NOT be plotted.

        cmap: str, optional, default: "vlag"
            The colormap to use for plotting the clustermap.

        title: str, optional, default: None
            The title of the heatmap plot.
            Note: If title is provided, title will be used as the name of
            the file when the figure is saved.

        dpi: int, optional, default: 600
            The amount of dots per inch to use when saving the figure. In
            accordance with Nature's guidelines, the default is 600.
            Source: https://www.nature.com/nature/for-authors/final-submission

        savefig: bool, optional, default: False
            When True, the plotted heatmap will be saved to the current working
            directory in pdf (IAW Nature's guidelines) format.
            Source: https://www.nature.com/nature/for-authors/final-submission
    """
    cmap = kwargs.get("cmap", "vlag")
    figsize = kwargs.get("figsize", (15, 15))
    title = kwargs.get("title", None)
    dendrograms = kwargs.get("dendrograms", True)

    cluster_map = sns.clustermap(dataframe.corr(), center=0, linewidths=.75, figsize=figsize, method="centroid", cmap=cmap)

    # Set the dendrograms in accordance with passed-in args
    cluster_map.ax_row_dendrogram.set_visible(dendrograms)
    cluster_map.ax_col_dendrogram.set_visible(dendrograms)

    if title:
        cluster_map.fig.suptitle(title)

    dpi = kwargs.get("dpi", 600)
    savefig = kwargs.get("savefig", False)
    if savefig:
        if title:
            filename = title
        else:
            filename = "my_seaborn_clustermap"

        plt.savefig("{}.pdf".format(filename), dpi=dpi)
