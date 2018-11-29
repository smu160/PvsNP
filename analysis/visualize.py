"""This module contains wrapper functions for plotting and visualizing data.

    @author: Saveliy Yusufov, Columbia University, sy2685@columbia.edu
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
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

        weights : array_like, shape(N,), optional, default: None
            An array of values ``w_i`` weighing each sample ``(x_i, y_i)``.
            Weights are normalized to 1 if `normed` is True. If `normed` is
            False, the values of the returned histogram are equal to the sum of
            the weights belonging to the samples falling into each bin.

        figsize: tuple, optional, default: (10, 10)
            The size of the heatmap plot.

        title: str, optional, default: 'Title Goes Here'
            The title of the heatmap plot.
            Note: The title will also be used as the name of the file when the
            figure is saved.

        dpi: int, optional, default: 600
            The amount of dots per inch to use when saving the figure. In
            accordance with Nature's guidelines, the default is 600.
            Source: https://www.nature.com/nature/for-authors/final-submission

        savefig: bool, optional, default: False
            When True, the plotted heatmap will be saved to the current working
            directory in pdf (IAW Nature's guidelines) format.
            Source: https://www.nature.com/nature/for-authors/final-submission
    """
    title = kwargs.get("title", "Title Goes Here")
    bins = kwargs.get("bins", (50, 50))
    weights = kwargs.get("weights", None)
    figsize = kwargs.get("figsize", (10, 10))

    heatmap, extent = generate_heatmap(x, y, sigma, bins=bins, weights=weights)
    plt.figure(figsize=figsize)
    plt.imshow(heatmap, origin="lower", extent=extent, cmap=cm.jet) # interpolation="gaussian"
    plt.title("{}".format(title))

    dpi = kwargs.get("dpi", 600)
    savefig = kwargs.get("savefig", False)
    if savefig:
        plt.savefig("{}.pdf".format(title), dpi=dpi);

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

    fig1, ax1 = plt.subplots(figsize=figsize)
    ax1.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=90)
    ax1.axis("equal")
    plt.show();
