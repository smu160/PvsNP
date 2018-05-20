# Hen Lab 

Signal processing and data analysis code to be used for the analysis of data collected during *in vivo* calcium imaging

### Why Python?
- [Eight Advantages of Python Over Matlab](http://phillipmfeldman.org/Python/Advantages_of_Python_Over_Matlab.html)
- [Interactive notebooks: Sharing the code](https://www.nature.com/news/interactive-notebooks-sharing-the-code-1.16261)
- [A gallery of interesting Jupyter Notebooks](https://github.com/jupyter/jupyter/wiki/A-gallery-of-interesting-Jupyter-Notebooks)

### TODO
#### Elevated Plus Maze Analysis: 
- [x] Figure out a definitive way to classify neuron selectivity 
- [x] Finish implementation of `is_neuron_selective()`
- [x] Write up documentation for all "cell selectivity" functions
- [x] *Significantly* improve the run-time of `shuffle()` (By parallelizing or coming up with better implementation)
- [x] Be able to calculate rates: Running, non running, open arms, closed arms for each animal, and for all cells combined (bar plot) and by animal (line plot)
- [x] Be able to handle varying frame rates for neuron or behavior data (want all files to be at 10fps for analysis)
- [x] Calculate rates for different time bins: (eg. first third of entire session, first minute of each behavior, first N entries in each arm)
- [x] Visualize correlation matrix with behavior data (concatenate activity and behavior dataframes)
- [x] Display correlation matrices for each behavior (eg. open arms vs closed arms)
- [x] Create pie charts to visualize the amount of neurons that were selective, not-selective, and unclassifiable for any behavior
- [x] Utilize real difference values to conclude what a selective neuron (after permutation test) is selective for exactly. i.e. behavior or $\neg$behavior
- [ ] generate plots of rates (AUC/sec and Events/sec)
- [ ] Plot scatter plot of neurons with open rates on the $y$-axis and closed rates on the $x$-axis, and $y=x$ line bisecting the scatter plot
- [ ] Plot the time-series for each neuron, vertically with way to distinguish behaviors
#### Graph Theoretical Analysis: 
- [x] Color nodes by their betweeness centrality
- [x] Plot the networks for different activities, e.g. What did a network of neurons look like when the mouse was doing a certain behavior
- [x] Create visualizations for all computed network measures
- [x] Implement and add mean clique size network measure 
- [ ] Color nodes by their selectivity in all graphs
- [ ] Plot all graphs by time
- [ ] Plot all graphs by periods of *continuous* behavior

For any other feature requests, feel free to email Saveliy Yusufov at sy2685@columbia.edu

## Getting Started

The best way to start using this code is by cloning the repository as follows:

1. Open a terminal on your machine
2. On the [main page](https://github.com/jaberry/Hen_Lab) of this repo, hit `Clone or download`
3. Copy the link in the small pop-up window that says `Clone with HTTPS`
4. In your terminal, enter the following line: `git clone <Web-URL>` (not including the `< >`)

### Prerequisites

In order to avoid dependency hell, it is highly reccomended that you install and use [Anaconda](https://www.anaconda.com/download/)

After installation, be sure to run `conda update --all` in your terminal.

If you already have Anaconda installed, then simply run `conda update --all` in your terminal.

You will need to [install the latest plotly package](https://anaconda.org/anaconda/plotly), as well. Open up your terminal or the command prompt and run: `conda install -c anaconda plotly`

In order to render plotly graphs, charts, and etc. in Jupyterlab, please follow the instructions provided [here](https://github.com/jupyterlab/jupyter-renderers/tree/master/packages/plotly-extension)

## Troubleshooting

If, for some reason, plotly graphs/charts used to render and they no longer render, try the following:

1. Open your terminal and run: `jupyter labextension uninstall @jupyterlab/plotly-extension`
2. After this command finishes executing, run: `jupyter labextension install @jupyterlab/plotly-extension`

This is only a band-aid solution, and the underlying issue will be explored (and hopefully fixed) in the near future.

For any issues, feel free to email Saveliy Yusufov at sy2685@columbia.edu

## Built With

* [pandas](http://pandas.pydata.org)
* [NetworkX](https://networkx.github.io)
* [seaborn](http://seaborn.pydata.org)
* [plotly](https://plot.ly)


## Authors

* **Jack Berry** - [jaberry](https://github.com/jaberry)
* **Saveliy Yusufov** - [smu160](https://github.com/smu160)


## Acknowledgments

* Jessica Jimenez 
