# Hen Lab 

Signal processing and data analysis code to be used for the analysis of data collected during *in vivo* calcium imaging

### Why Python?
- [Eight Advantages of Python Over Matlab](http://phillipmfeldman.org/Python/Advantages_of_Python_Over_Matlab.html)
- [Interactive notebooks: Sharing the code](https://www.nature.com/news/interactive-notebooks-sharing-the-code-1.16261)
- [A gallery of interesting Jupyter Notebooks](https://github.com/jupyter/jupyter/wiki/A-gallery-of-interesting-Jupyter-Notebooks)

### TODO
#### Elevated Plus Maze Analysis: 
- [x] QA cell selectivity 
- [x] Plot pie charts of cell selectivity by each animal and all together
- [ ] Compute rates per entry into open arms and closed arms
- [ ] Plot line charts of rates by entry and time
- [ ] Create demo of cell selectivity
#### Graph Theoretical Analysis: 
- [ ] Plot all the cells as they are physically
- [ ] Plot graphs of rearing, freezing, and neither within each of the 3 days.
- [ ] Analyze NO, OFT, POPP data for DRD87
#### Create Powerpoint:
- [ ] Cell selectivity and rates for all 5 animals
- [ ] Pie charts of neuron classification for all 5 animals
- [ ] Line graphs of by entry rates
- [ ] Overall graph
- [ ] Overall graph colored by selectivity 
- [ ] Overall graphs of open vs. closed
- [ ] Overall graphs of open vs. closed colored by selectivity
- [ ] All network measures (leave out uninteresting measures)

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
