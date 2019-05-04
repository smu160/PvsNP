PvsNP
=======
[![DOI](https://zenodo.org/badge/166607518.svg)](https://zenodo.org/badge/latestdoi/166607518)

<img src="python_action_potential.png" width="500" align="right">

<b>P</b>ython <b>vs</b>. <b>N</b>euro-<b>P</b>hysiological (PvsNP) serves as a toolbox for:
* Statistical analysis of neurophysiological data, including but not limited to cell selectivity, place cell analysis, and etc.
* Graph Theoretical Analysis of networks of neurons.
* [Data Visualization tool](https://github.com/smu160/PvsNP/tree/master/gui) - see the "full picture" of the data garnered from experiments with integration of calcium imaging movies, behavioral videos, and data processing results.
* Deconvolution of calcium imaging data. (Deprecated)

For any feature requests, feel free to [create an issue](https://help.github.com/articles/creating-an-issue/).

<sub><sup>[image source](https://github.com/nipy/nipy-artwork/blob/master/pics/python_action_potential.svg)</sup></sub>

## Getting Started

### Docker

1. Download and install [Anaconda](https://docs.anaconda.com/anaconda/install/) (Python 3.X)

2. Download and install [Docker](https://www.docker.com/get-started)

3. Clone the repository:
```bash
git clone https://github.com/smu160/PvsNP.git
```

3. Navigate into your local repository and build the Docker image:
```bash
cd PvsNP
docker build . -t jupyter
```

4. Use the image to run a container:
```bash
docker run -it -p 8888:8888 jupyter
```

If you need to mount data to the container, then use the following command:
```bash
docker run -it -p 8888:8888 -v source_directory:target_directory jupyter
```

5. You should see something along the lines of:
```bash
...
    Or copy and paste one of these URLs:
        http://(2958ngdf42t or 127.0.0.1):8888/?token=XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
```

Now open your browser window and go to the URL that was created for you.

### Installation on Mac or Linux (Python 3.x)

1. Download and install [Anaconda](https://docs.anaconda.com/anaconda/install/) (Python 3.X)

2. Clone the repository:
```bash
git clone https://github.com/smu160/PvsNP.git
```

3. Navigate into your local repository and create your environment:
```bash
cd PvsNP
bash create_env.sh
```

## Troubleshooting

[Create an issue](https://help.github.com/articles/creating-an-issue/)

## Dependencies

A list of dependencies can be found in the [environment file](https://github.com/smu160/PvsNP/blob/master/environment.yml)
