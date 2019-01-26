Hen Lab
======
# <img src="readme_pic.jpg">

This code base was written with the intention of serving as a toolbox for:

1. Statistical Testing of neuron classification by behavior.
2. Graph Theoretical Analysis of networks of neurons.
3. Data Visualization tool -- see the "full picture" of the data garnered from experiments with integration of calcium imaging movies, behavioral videos, and data processing results.
4. Deconvolution of calcium imaging data. (Deprecated)

For any feature requests, [create an issue](https://help.github.com/articles/creating-an-issue/).

## Getting Started

### Docker

1. Download and install [Anaconda](https://docs.anaconda.com/anaconda/install/) (Python 3.X)

2. Download and install [Docker](https://www.docker.com/get-started)

3. Clone the repository:
```bash
git clone https://github.com/jaberry/Hen_Lab.git
```

3. Navigate into your local repository and build the Docker image:
```bash
cd Hen_Lab
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
git clone https://github.com/jaberry/Hen_Lab.git
```

3. Navigate into your local repository and create your environment:
```bash
cd Hen_Lab
bash create_env.sh
```

## Troubleshooting

[Create an issue](https://help.github.com/articles/creating-an-issue/)

For members of the Hen Lab, feel free to message Saveliy Yusufov via Slack.


## Why Python?
- [Eight Advantages of Python Over Matlab](http://phillipmfeldman.org/Python/Advantages_of_Python_Over_Matlab.html)
- [Interactive notebooks: Sharing the code](https://www.nature.com/news/interactive-notebooks-sharing-the-code-1.16261)
- [A gallery of interesting Jupyter Notebooks](https://github.com/jupyter/jupyter/wiki/A-gallery-of-interesting-Jupyter-Notebooks)
- [Food for Thought](https://www.theatlantic.com/science/archive/2018/04/the-scientific-paper-is-obsolete/556676/)
