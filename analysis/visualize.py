import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.ndimage.filters import gaussian_filter

def generate_heatmap(x, y, sigma, bins=(50, 50)):
    """Generates a heatmap

    Args:
        x: array

        y: array

        sigma: int

        bins: tuple, optional

    Returns:
        heatmap:

        extent:
    """

    heatmap, xedges, yedges = np.histogram2d(x, y, bins=bins)
    heatmap = gaussian_filter(heatmap, sigma=sigma)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    return heatmap.T, extent

def plot_heatmap(x, y, sigma, **kwargs):
    """ Plots a heatmap

    Wrapper function for matplotlib, numpy, and scipy that
    plots a heatmap with the capability to smooth the heatmap.

    Args:
        x: array

        y: array

        sigma: int

        bins: tuple, optional

        figsize: tuple, optional

        title: str, optional

        dpi: int, optional
    """
    title = kwargs.get("title", "Title Goes Here")
    bins = kwargs.get("bins", (50, 50))

    heatmap, extent = generate_heatmap(x, y, sigma, bins=bins)
    plt.figure(figsize=kwargs.get("figsize", (10, 10)))
    plt.imshow(heatmap, origin="lower", cmap=cm.jet)
    plt.title("{}".format(title))
    plt.axis("off")

    dpi = kwargs.get("dpi", 300)
    if kwargs.get("savefig", False):
        plt.savefig("{}.png".format(title), dpi=dpi);

def pie_chart(sizes, *labels, **kwargs):
    """Wrapper method for matplotlib's pie chart.

    The slices will be ordered and plotted counter-clockwise.

    Args:
        sizes: list
            A list of the sizes of each category.

        labels: str, variable number of args
            The label for each corresponding slice size.

        figsize: tuple, optional
            The size of the figure to be plotted; default is (5,5).
    """
    if len(labels) != len(sizes):
        raise ValueError("Length of sizes and amount of labels must be equal.")

    figsize = kwargs.get("figsize", (5,5))
    fig1, ax1 = plt.subplots(figsize=figsize)
    ax1.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=90)
    ax1.axis('equal')
    plt.show();


