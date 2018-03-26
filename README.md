# Hen Lab 

Signal processing and data analysis code to be used for the analysis of data collected during *in vivo* calcium imaging

### Why Python?
- [Eight Advantages of Python Over Matlab](http://phillipmfeldman.org/Python/Advantages_of_Python_Over_Matlab.html)
- [Interactive notebooks: Sharing the code](https://www.nature.com/news/interactive-notebooks-sharing-the-code-1.16261)
- [A gallery of interesting Jupyter Notebooks](https://github.com/jupyter/jupyter/wiki/A-gallery-of-interesting-Jupyter-Notebooks)

### TODO
#### Elevated Plus Maze Analysis: 
- [ ] Figure out a definitive way to classify neuron selectivity
- [x] Finish implementation of `is_neuron_selective()`
- [ ] Add functionality to run `is_neuron_selective()` based on different periods of time
- [ ] Write up documentation for all "cell selectivity" functions
- [ ] Move all functions from EPM Analysis notebook to `analysis_utils.py` once they are complete and sound
- [x] *Significantly* improve the run-time of `shuffle()` (By parallelizing or coming up with better implementation)
#### Graph Theoretical Analysis: 
- [ ] Color nodes by their selectivity in all graphs
- [x] Plot the networks for different activities, e.g. What did a network of neurons look like when the mouse was doing a certain behavior
- [ ] Implement a function that will find all time periods of some (continuous) behavior 
- [ ] Create visualizations for all computed network measures

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

In order to render plotly graphs, charts, and etc., please follow the instructions provided [here](https://github.com/jupyterlab/jupyter-renderers/tree/master/packages/plotly-extension)

## Troubleshooting

If, for some reason, plotly graphs/charts used to render and they no longer render, try the following:

1. Open your terminal and run: `jupyter labextension uninstall @jupyterlab/plotly-extension`
2. After this command finishes executing, run: `jupyter labextension install @jupyterlab/plotly-extension`

This is only a band-aid solution, and the underlying issue will be explored (and hopefully fixed) in the near future.

For any issues, feel free to email Saveliy Yusufov at sy2685@columbia.edu

## Built With

* [Python](https://www.python.org)
* [pandas](http://pandas.pydata.org)
* [NetworkX](https://networkx.github.io)
* [seaborn](http://seaborn.pydata.org)
* [plotly](https://plot.ly)

## Contributing

Please read [...] for details on our code of conduct, and the process for submitting pull requests to us.

## Versioning

...

## Authors

* **Jack Berry** - [jaberry](https://github.com/jaberry)
* **Saveliy Yusufov** - [smu160](https://github.com/smu160)

## License

...

## Acknowledgments

* Jessica Jimenez 
