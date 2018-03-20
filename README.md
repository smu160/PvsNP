# Hen Lab 

Signal processing and data analysis code to be used for the analysis of data collected during *in vivo* calcium imaging

## Getting Started

The best way to start using this code is by cloning the repository as follows:

1. Open a terminal on your machine
2. On the [main page](https://github.com/jaberry/Hen_Lab) of this repo, hit `Clone or download`
3. Copy the link in the small pop-up windows that says `Clone with HTTPS`
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

For any issues or feature requests, feel free to email Saveliy Yusufov at sy2685@columbia.edu

## Built With

* [Python](https://www.python.org)
* [pandas](http://pandas.pydata.org)
* [NetworkX](https://networkx.github.io)
* [seaborn](http://seaborn.pydata.org)
* [plotly](https://plot.ly)

## Contributing

TODO

Please read [CONTRIBUTING.md] for details on our code of conduct, and the process for submitting pull requests to us.

## Versioning

TODO

## Authors

* **Jack Berry** - [jaberry](https://github.com/jaberry)
* **Saveliy Yusufov** - [smu160](https://github.com/smu160)

## License

TODO

## Acknowledgments

TODO

* Jessica Jimenez 
