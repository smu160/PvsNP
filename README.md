# Hen Lab 

Signal processing and data analysis code to be used for the analysis of data collected during *in vivo* calcium imaging

### Why Python?
- [Eight Advantages of Python Over Matlab](http://phillipmfeldman.org/Python/Advantages_of_Python_Over_Matlab.html)
- [Interactive notebooks: Sharing the code](https://www.nature.com/news/interactive-notebooks-sharing-the-code-1.16261)
- [A gallery of interesting Jupyter Notebooks](https://github.com/jupyter/jupyter/wiki/A-gallery-of-interesting-Jupyter-Notebooks)

For any feature requests, feel free to email Saveliy Yusufov at sy2685@columbia.edu

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

Running the resmpling functions in analysis/resampling.py will only work on 
Unix based operating systems, i.e. Linux, MacOS, and etc. This issue will be 
addressed with the creation of a Dockerfile for those of you who are using 
Windows. In the mean time, you have two choices: 

1. Use a machine that has a Unix based operating system.
2. Install and use any Linux distro of your choice. I reccomend [Ubuntu](https://www.ubuntu.com/download/desktop)


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
