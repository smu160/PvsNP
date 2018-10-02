import matplotlib.pyplot as plt

def pie_chart(sizes, *labels, **kwargs):
    """wrapper method for matplotlib's pie chart

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


